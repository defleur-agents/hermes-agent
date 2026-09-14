# gateway/ — messaging gateway, adapters, delivery

Applies on top of the root `AGENTS.md`. Long-form: `website/docs/developer-guide/gateway-internals.md`.
New platform adapter: follow `gateway/platforms/ADDING_A_PLATFORM.md` step by step.

## Shape

`gateway/run.py` is the facade; phases live in `run_*.py` (startup, adapters, inbound, turn, busy,
goals, notifications, shutdown, ...), sessions in `session*.py`, slash handlers in
`slash_commands_*.py` mixins, authorization in `authz_mixin.py`, adapters in `platforms/<name>.py`
over `platforms/base.py`. `builtin_hooks/` is the extension point for always-registered gateway
hooks (none shipped). The gateway reads user YAML **raw** (`run.py` + `config.py`), not through
`DEFAULT_CONFIG` — a key the CLI sees but the gateway doesn't means you're on the wrong loader
(`hermes_cli/AGENTS.md`). Each adapter picks a base toolset (Telegram → `"messaging"`).

Slash commands: handlers are looked up by name through `_command_handler_table`; a command is
listed in `_IDLE_COMMANDS` or `_PLAIN_COMMANDS` (works mid-run) in `run_busy.py`. No
`if canonical == ...` chains. Registry + adding a command: `hermes_cli/AGENTS.md`.

## The gateway has TWO message guards — both must bypass approval/control commands

While an agent is running, an inbound message passes two sequential guards: (1) the **base
adapter** (`platforms/base.py`) queues it in `_pending_messages` when `session_key in
self._active_sessions`; (2) the **runner** (`run.py`/`run_busy.py`) intercepts `/stop`, `/new`,
`/queue`, `/status`, `/approve`, `/deny` before they reach `running_agent.interrupt()`. Any new
command that must reach the runner while the agent is blocked (approval prompts) MUST bypass BOTH
guards and be dispatched inline — never via `_process_message_background()`, which races session
lifecycle.

## Streaming delivery contract (stream-is-the-message adapters)

Adapters with `draft_stream_is_message = True` (relay Slack native streaming) keep ONE cumulative
native stream per turn; the stream IS the final message. Four invariants, each from a live
duplicate-final incident (NS-658 canary ledger, hermes#85796 / gateway-gateway#210); violating any
re-creates a duplicate or frozen stream:

1. **Draft frames are prefix-stable.** Frame N must be a string prefix of frame N+1. Never mutate
   drafts per tick — no fence-closing (`ensure_closed_code_fences`), cursor suffix, segment-state
   resets at tool boundaries, or mrkdwn conversion. A non-prefix frame forces a whole-snapshot
   re-append ("stacked copies"). The finalize path may still transform the real final.
2. **The consumer declares the final; the adapter never guesses.** `finish(final_text)` carries the
   completed `final_response` (verifier footer, completion explainer included). New post-stream
   augmentation MUST ride this payload — mutating `final_response` after the seal re-opens the
   `delivered_final_matches` mismatch → corrective duplicate send.
3. **Interim sends carry `metadata["_interim_send"] = True`.** Any consumer-side `adapter.send()`
   that is not the turn-final (commentary, segment-tail flushes) must set it or seal-interception
   seals the live stream with interim text. Seal-interception exists at BOTH egress doors (`send()`
   and `send_for_platform()`); a new egress door needs the same two checks.
4. **Reconcile by edit, never by plain send.** A lane delivering a final beside a sealed stream
   (queued follow-ups, media-accompanied finals) first tries `edit_message` on the consumer's
   `message_id`; plain `send()` only when no editable message exists. A sealed native stream is a
   regular message — `chat.update` works (live-verified).

Contract tests: `tests/gateway/test_stream_final_contract.py` (mutation-checked). Slack ground
truth: `chat.*Stream` speaks STANDARD markdown, not mrkdwn; `stopStream.markdown_text` APPENDS;
`startStream`/`stopStream` are Tier 2 (~20/min). Check `draft_stream_is_message is True` —
MagicMock adapters in older tests auto-create truthy attributes.

## Background process notifications

`terminal(background=true, notify_on_complete=true)` starts a gateway watcher that detects
completion and triggers a new agent turn. Verbosity: `display.background_process_notifications`
(or `HERMES_BACKGROUND_NOTIFICATIONS`): `concise` (default; one line, failures append an output
tail), `all` (running updates + final raw output), `result` (final raw output only), `error`
(final raw output only on non-zero exit), `off`.

The idle completion watcher also drains `watch_match` / `watch_disabled`; no user follow-up is
required. Notify-off drains these without waking. Transport failures are retried; unavailable
durable completion owners/transports do not spend delivery attempts. Profile-namespaced process
events keep their owning adapter, including raw progress/final notices. Adapter acceptance is
at-least-once admission, not proof that a model turn or final outbound reply completed. API-server
async completions remain durable delivery rows, never autonomous new model turns. Require the
`admit_internal_event` receipt for completion/watch injection: a handler returning None is not
acceptance. Refused admission refunds every claimed batch sibling without spending an attempt;
actual delivery errors keep their bounded retry policy. Recognized raw API routes resolve after
persisted messaging origins and defer quietly when unavailable; malformed routes still warn.

### Durable completions on finite API/Wizard runs

A `/v1/runs` turn ends with its HTTP response, so a detached `notify_on_complete` worker it spawned
has no live parent. `session_context.completion_delivery_mode()` declares the durable route home;
`APIServerAdapter` binds `api_session_wake` only for an authenticated session whose history the
server owns and can resume (same #98619 audit as `session_history_delivery`), and advertises it as
the `background_process_completion` capability. The key it checks is `_expected_api_key()`, the one
authorized on THIS request's route, so a named profile with no key of its own promises nothing
rather than promising a wake that would borrow the owner's. The terminal tool validates that route
BEFORE spawning and fails closed on anything it cannot honor: no route, or watch-pattern streaming
(which needs an authorized turn per match where the route grants one, for the terminal outcome),
means no PID and no process entry rather than a promise nothing can keep. Other finite surfaces
(one-shot CLI, cron, Kanban) keep their "poll it yourself" downgrade.

An armed spawn writes its route into the first checkpoint AND the first receipt
(`tools/process_registry_outbox.py`, receipt schema 2), so an immediate exit cannot produce a
routeless completion; the exit settles it to `pending` before the session leaves the running set.
Arming is a hard precondition, not best effort: if the obligation cannot be written the registry
reaps the child it just spawned and the tool call fails. The `APIServerAdapter` owns the replay
loop (started from `connect()`, cancelled on bind failure and on `disconnect()`, so a standalone
`wizard-api.service` works without a GatewayRunner): it claims only its own receipts, reconciles
armed receipts whose worker is gone into pending unknown outcomes, formats with
`format_process_notification()` and self-posts through `gateway.wake`, settling only after an
accepted delivery AND a successful settle. The outbox claim is the SINGLE arbiter — a
durable-route completion is deliberately not queued on `completion_queue` and registers no gateway
watcher. `wait`/`log`/explicit kill settle it `consumed`, and never by stealing a live claim: on a
claimed receipt the intent is recorded and the claim holder converges it, so no completion can
produce a second turn. Schema-1 receipts never replay. Route resolution prefers
`origin_session_id`, then the API `chat_id`, and only then `session_key` — on `/v1/runs` the
session key is the per-run approval id and resumes no conversation.

Receipts are profile-LOCAL (they live in the spawning profile's `HERMES_HOME`) and carry the
spawning profile as immutable identity, so under `multiplex_profiles` one replay pass enumerates
every served profile's outbox and wakes each on its own `/p/<profile>/v1/chat/completions` mirror
with that profile's own `API_SERVER_KEY`. A profile whose key is unresolvable is held, never
delivered under another principal's credential.

Exactly-once survives a restart. The per-completion `Idempotency-Key` is checked against a DURABLE
ledger (`reserve_wake_delivery`) kept in the receipt the key names, not only against the API
server's in-memory `_IdempotencyCache` — so a wake accepted by a process that died before settling
its claim is replayed by the next process instead of re-running the user's turn. A claim is
exclusive but expires, so a crashed consumer's receipt becomes replay-eligible again after
`CLAIM_TTL_SECONDS` and converges to an observable `dropped` once the attempt budget is spent.
Unresolved receipts (armed/pending/claimed) are never pruned by age or by the retained-result
count cap; only a terminal one is collectable, and its `.lock` sidecar is unlinked solely from
inside that lock.

Cron execution has its own session. Eligible continuable deliveries may mirror or seed the
reply-facing conversation: origin, origin-less home fallback, user-written bare-platform home,
or opted-in explicit targets. `all` expansions do not gain home mirror eligibility. Mirrored
briefs are labelled user turns appended at a turn boundary, preserving role alternation
(`cron/AGENTS.md`).

## `/login` (off-turn, paired DM only)

`/login` is registered in `hermes_cli/commands.py` with `busy_policy="dispatch"` and
`desktop="settings"`, listed in `run_busy.py::_PLAIN_COMMANDS`, and handled by
`GatewayLoginCommandsMixin` (`gateway/slash_commands_login.py`). It refuses outside a paired DM:
`chat_type in {"dm","private"}`, a truthy `chat_id`, and a platform whose `"dm"` really is a paired
conversation — ntfy, raft and a2a all report `chat_type="dm"` for a broadcast topic, a channel and
an agent peer, so posting a consent link there would publish it.

**It binds the whole install.** The sign-in writes the singleton `providers.nous`, so whoever
approves the code owns this gateway's inference and connectors for every chat it serves. Slash
gating is opt-in (`gateway/slash_access.py`): with no `allow_admin_from` set, every user allowed to
DM the bot can run it. Operators of shared gateways must set it.

The handler returns its ack at once and drains `anon_auth.run_sign_in` on a **private single-worker
executor** (`_login_executor`), never the shared 10-thread gateway pool — a promotion wait can last
the code's full expiry, and a live worker in the shared pool makes shutdown skip the SessionDB
close/checkpoint. One attempt per process, stamped with the identity that started it: the same
identity's second `/login` supersedes (the loser ends with the superseded copy pushed into its own
chat); a different identity is refused. `task.cancel()` cannot interrupt a blocking poll inside a
worker thread, so shutdown and supersede both work through `attempt.cancelled`, which
`wait_for_promotion` now polls on a ≤1 s tick. A shutdown-cancelled attempt pushes nothing — the
task is dying and its adapters may already be gone — but if the server had already completed the
transfer it still persists, because that transfer is irreversible.

On completion the handler **evicts** every cached agent still on `nous/welcome` and clears any
session model override pinned to it. It does not switch agents in place and does not write a model
override: `settle_after_upgrade` already moved `model.default`/`model.base_url` in the config, every
turn re-resolves the config and the credentials, and `_agent_config_signature` already forces a
rebuild when the route changes — so an in-place swap buys no cache warmth and an override would pin
an expiring access token that nothing refreshes.

## Gateway lifecycle vs. the Desktop app

`hermes serve` (control plane, desktop-spawned child) dies with the app — by design. The messaging
gateway (`gateway run`) SURVIVES the app: the serve backend's `/api/gateway/*` endpoints spawn it
detached (`_spawn_hermes_action` — `start_new_session` / `DETACHED_PROCESS`), so `before-quit`'s
SIGTERM never reaches it and bots keep running. The known breach is the Windows shim-unlock
teardown (`taskkill /T /F` on venv-shim holders, #85265), which exists to let updates proceed and is
replaced by #92091's `pause-for-update`. Do NOT "fix" gateway-dies-with-app by re-parenting the
gateway under the backend, and do NOT "fix" update locks by widening the tree-kill. Gateways stamp
`code_sha`/`code_version` into `gateway_state.json` (`status.py`) so the updater can verify a fleet.

## Profiles and secrets in adapters

- **Token locks.** An adapter that connects with a unique credential (bot token, API key) calls
  `acquire_scoped_lock()` from `gateway.status` in `connect()`/`start()` and `release_scoped_lock()`
  in `disconnect()`/`stop()`, so two profiles cannot share one credential. Canonical:
  `plugins/platforms/irc/adapter.py`.
- **Multiplex profile-scoped env reads MUST fail closed — never borrow from `os.environ`**
  (`agent/secret_scope.py`; #72348, #86905). Under `gateway.multiplex_profiles`, `os.environ` holds
  the DEFAULT profile's values; a secondary profile's `.env` exists only in its secret scope,
  installed per turn by `_profile_runtime_scope`. All profile-level env config — credentials
  (`app_secret`, tokens) AND authorization (`FEISHU_ALLOWED_USERS`, `{PLATFORM}_ALLOW_ALL_USERS`,
  `GATEWAY_ALLOW_ALL_USERS`, `group_policy`, `allow_bots`) — is read scope-aware: adapters via
  `_get_scoped_secret()` (canonical fail-closed copy: `plugins/platforms/feishu/adapter.py`),
  gateway authz via `_auth_env()` / `_platform_gate_env()` (`authz_mixin.py`). Scope installed +
  multiplex active → a scoped miss returns the **default**, NEVER `os.environ` (a leaked allowlist
  skips the allow-all check and silently rejects every secondary-profile sender, #86905). The
  unscoped default-profile path (`UnscopedSecretError`) and single-profile deployments keep the
  `os.environ` read — there it IS the profile's own value. `_get_scoped_secret` is copy-pasted
  across ~15 adapters: when touching one, verify fail-closed semantics and never reintroduce the
  `except _UnscopedSecretError: val = os.getenv(...)` fallback-after-miss shape.

## Tests

`tests/gateway/`. Adapter tests exercise the real delivery path with a fake transport; never
assert on hardcoded platform lists or command counts (root: no change-detectors). Session-key and
guard behaviour are invariants worth a test; platform API quirks belong in connector comments +
tests, not in prose.
