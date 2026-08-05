"""Behavior contracts for James's managed title/topic customizations."""

from unittest.mock import MagicMock, patch

from agent.title_generator import (
    auto_title_session,
    choose_topic_icon,
    generate_title,
    maybe_auto_title,
    maybe_auto_title_from_opener,
)
from hermes_state import SessionDB


class TestManagedTitleAliases:
    def test_compact_preferences_and_name_aliases_are_configurable(self):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Planning Flow"
            config = {
                "auxiliary": {
                    "title_generation": {
                        "max_words": 2,
                        "max_characters": 24,
                        "name_aliases": {
                            "project atlas": "ProjectAtlas",
                            "atlas app": "ProjectAtlas",
                        },
                    }
                }
            }

            with (
                patch("hermes_cli.config.load_config_readonly", return_value=config),
                patch("agent.title_generator.call_llm", return_value=mock_response) as llm,
            ):
                assert generate_title("Update the ATLAS APP", "I will inspect it") == "ProjectAtlas"

            system_prompt = llm.call_args.kwargs["messages"][0]["content"]
            assert "1-2 words" in system_prompt
            assert "24 characters" in system_prompt
            assert '\"project atlas\": \"ProjectAtlas\"' in system_prompt
            assert '\"atlas app\": \"ProjectAtlas\"' in system_prompt

    def test_creative_style_and_user_instructions_are_configurable(self):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Commerce Cockpit"
            config = {
                "auxiliary": {
                    "title_generation": {
                        "style": "creative",
                        "instructions": "Favor concrete systems imagery.",
                    }
                }
            }

            with (
                patch("hermes_cli.config.load_config_readonly", return_value=config),
                patch("agent.title_generator.call_llm", return_value=mock_response) as llm,
            ):
                assert generate_title("Build an Amazon agent", "Let's map it") == "Commerce Cockpit"

            system_prompt = llm.call_args.kwargs["messages"][0]["content"]
            assert "vivid, memorable" in system_prompt
            assert "Concrete metaphors" in system_prompt
            assert "Favor concrete systems imagery." in system_prompt

    def test_collision_retry_prompt_rejects_numbered_variants(self):
            responses = []
            for title in ("Hermes", "Amazon Agent"):
                response = MagicMock()
                response.choices = [MagicMock()]
                response.choices[0].message.content = title
                responses.append(response)

            with patch("agent.title_generator.call_llm", side_effect=responses) as llm:
                assert generate_title("Compare Claude with Hermes", "Let's evaluate it") == "Hermes"
                assert generate_title(
                    "Compare Claude with Hermes",
                    "Let's evaluate it",
                    avoid_titles=["Hermes"],
                ) == "Amazon Agent"

            retry_prompt = llm.call_args_list[1].kwargs["messages"][0]["content"]
            assert '\"Hermes\"' in retry_prompt
            assert "Do not append a number" in retry_prompt

    def test_name_aliases_match_whole_terms_not_substrings(self):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Atlassian Migration"
            config = {
                "auxiliary": {
                    "title_generation": {
                        "name_aliases": {"atlas": "ProjectAtlas"},
                    }
                }
            }

            with (
                patch("hermes_cli.config.load_config_readonly", return_value=config),
                patch("agent.title_generator.call_llm", return_value=mock_response),
            ):
                title = generate_title("Plan the Atlassian migration", "I will inspect it")

            assert title == "Atlassian Migration"

    def test_name_aliases_do_not_match_across_exchange_boundary(self):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Vajra"
            config = {
                "auxiliary": {
                    "title_generation": {
                        "name_aliases": {
                            "atlas app": "ProjectAtlas",
                            "vajra": "Vajra",
                        },
                    }
                }
            }

            with (
                patch("hermes_cli.config.load_config_readonly", return_value=config),
                patch("agent.title_generator.call_llm", return_value=mock_response),
            ):
                title = generate_title(
                    "Please leave this as atlas",
                    "app support for Vajra is available",
                )

            assert title == "Vajra"

    def test_canonical_alias_takes_precedence_over_character_preference(self):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Atlas"
            config = {
                "auxiliary": {
                    "title_generation": {
                        "max_characters": 12,
                        "name_aliases": {"atlas app": "ProjectAtlasLongName"},
                    }
                }
            }

            with (
                patch("hermes_cli.config.load_config_readonly", return_value=config),
                patch("agent.title_generator.call_llm", return_value=mock_response),
            ):
                title = generate_title("Open the atlas app", "I will inspect it")

            assert title == "ProjectAtlasLongName"

    def test_collision_retry_can_qualify_a_canonical_alias(self):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "ProjectAtlas Roadmap"
            config = {
                "auxiliary": {
                    "title_generation": {
                        "name_aliases": {"atlas app": "ProjectAtlas"},
                    }
                }
            }

            with (
                patch("hermes_cli.config.load_config_readonly", return_value=config),
                patch("agent.title_generator.call_llm", return_value=mock_response),
            ):
                title = generate_title(
                    "Open the atlas app roadmap",
                    "I will inspect it",
                    avoid_titles=["ProjectAtlas"],
                )

            assert title == "ProjectAtlas Roadmap"

    def test_name_alias_after_prompt_snippet_is_still_enforced(self):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Planning Flow"
            config = {
                "auxiliary": {
                    "title_generation": {
                        "name_aliases": {"atlas app": "ProjectAtlas"},
                    }
                }
            }

            with (
                patch("hermes_cli.config.load_config_readonly", return_value=config),
                patch("agent.title_generator.call_llm", return_value=mock_response),
            ):
                title = generate_title("x" * 550 + " atlas app", "I will inspect it")

            assert title == "ProjectAtlas"

    def test_configured_character_limit_is_enforced(self):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "A" * 40
            config = {
                "auxiliary": {
                    "title_generation": {
                        "max_words": 2,
                        "max_characters": 24,
                    }
                }
            }

            with (
                patch("hermes_cli.config.load_config_readonly", return_value=config),
                patch("agent.title_generator.call_llm", return_value=mock_response),
            ):
                title = generate_title("question", "answer")

            assert title is not None
            assert len(title) == 24
            assert title.endswith("...")

    def test_rejected_alias_long_response_respects_character_limit(self):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "A" * 40
        config = {
            "auxiliary": {
                "title_generation": {
                    "max_characters": 24,
                    "name_aliases": {"atlas app": "ProjectAtlas"},
                }
            }
        }

        with (
            patch("hermes_cli.config.load_config_readonly", return_value=config),
            patch("agent.title_generator.call_llm", return_value=response),
        ):
            title = generate_title(
                "Open the atlas app roadmap",
                "I will inspect it",
                avoid_titles=["ProjectAtlas"],
            )

        assert title is not None
        assert len(title) == 24
        assert title.endswith("...")


class TestManagedTopicIconSelection:
    def test_returns_exact_allowed_emoji(self):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "🚀"

            with patch("agent.title_generator.call_llm", return_value=mock_response) as llm:
                result = choose_topic_icon(
                    "ProjectAtlas",
                    "Let's improve the ProjectAtlas planning workflow",
                    ["📊", "🚀", "🛠️"],
                )

            assert result == "🚀"
            prompt = llm.call_args.kwargs["messages"][0]["content"]
            assert "ranked" in prompt
            assert "specific, playful visual metaphors" in prompt
            assert "📊" in prompt and "🚀" in prompt and "🛠️" in prompt
            assert llm.call_args.kwargs["temperature"] == 0.7

    def test_extracts_single_allowed_emoji_from_wrapped_response(self):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Emoji: 💳"

            with patch("agent.title_generator.call_llm", return_value=mock_response):
                assert choose_topic_icon("Finance", "credit card benefits", ["🚀", "💳"]) == "💳"

    def test_matches_allowed_variation_selector_form(self):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "⚡"

            with patch("agent.title_generator.call_llm", return_value=mock_response):
                assert choose_topic_icon("ProjectBolt", "fast analysis", ["⚡️", "💡"]) == "⚡️"

    def test_prefers_ranked_candidate_that_was_not_used_recently(self):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "🚀 📊 🧪"

            with patch("agent.title_generator.call_llm", return_value=mock_response) as llm:
                assert choose_topic_icon(
                    "Launch",
                    "compare metrics",
                    ["🚀", "📊", "🧪"],
                    recent_emojis=["🚀"],
                ) == "📊"

            prompt = llm.call_args.kwargs["messages"][0]["content"]
            assert "used recently" in prompt
            assert "🚀" in prompt

    def test_excludes_recent_icons_from_large_candidate_pool(self):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "💻"

            with patch("agent.title_generator.call_llm", return_value=mock_response):
                assert choose_topic_icon(
                    "Developer Tools",
                    "debug the agent",
                    ["💻", "🎨", "🧪", "🔭", "🛠️"],
                    recent_emojis=["💻"],
                ) is None

    def test_compound_emoji_wins_over_overlapping_component(self):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "👮‍♂️ ⚡"

            with patch("agent.title_generator.call_llm", return_value=mock_response):
                assert choose_topic_icon(
                    "Safety",
                    "police safety workflow",
                    ["👮", "👮‍♂️", "♂️", "⚡️"],
                ) == "👮‍♂️"
                assert choose_topic_icon(
                    "Safety",
                    "police safety workflow",
                    ["👮", "👮‍♂️", "♂️", "⚡️"],
                    recent_emojis=["👮‍♂️"],
                ) == "⚡️"

    def test_falls_back_to_top_ranked_candidate_when_all_were_recent(self):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "💻 🤖"

            with patch("agent.title_generator.call_llm", return_value=mock_response):
                assert choose_topic_icon(
                    "Developer Tools",
                    "debug the agent",
                    ["💻", "🤖"],
                    recent_emojis=["💻", "🤖"],
                ) == "💻"

    def test_rejects_response_without_an_allowed_candidate(self):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "🛸"

            with patch("agent.title_generator.call_llm", return_value=mock_response):
                assert choose_topic_icon("Launch", "ship it", ["🚀", "📊"]) is None

    def test_skips_without_allowed_icons(self):
            with patch("agent.title_generator.call_llm") as llm:
                assert choose_topic_icon("Launch", "ship it", []) is None
            llm.assert_not_called()


class TestManagedOpenerTitles:
    def test_opener_fires_before_first_response(self):
            """Topic lanes can title from the opener while the main turn is running."""
            db = MagicMock()
            db.get_session_title.return_value = None

            import threading

            called = threading.Event()
            with patch("agent.title_generator.auto_title_session") as mock_auto:
                mock_auto.side_effect = lambda *a, **k: called.set()
                maybe_auto_title_from_opener(
                    db,
                    "sess-opener",
                    "audit the last fourteen memory reports",
                    [],
                    main_runtime={},
                )
                assert called.wait(timeout=10), "opener auto-title thread never ran"
                mock_auto.assert_called_once_with(
                    db,
                    "sess-opener",
                    "audit the last fourteen memory reports",
                    "",
                    failure_callback=None,
                    main_runtime={},
                    title_callback=None,
                    runtime_validator=None,
                )

    def test_opener_skips_after_meaningful_assistant_response(self):
            db = MagicMock()
            db.get_session_title.return_value = None
            history = [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "completed response"},
            ]

            with patch("agent.title_generator.auto_title_session") as mock_auto:
                maybe_auto_title_from_opener(db, "sess-old", "follow up", history)

            mock_auto.assert_not_called()

    def test_image_burst_titles_on_first_completed_response(self):
            """Interrupted image rows must not consume the first-exchange window."""
            db = MagicMock()
            db.get_session_title.return_value = None
            history = [
                {"role": "user", "content": "first screenshot"},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": "second screenshot"},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": "explain the screenshots"},
                {"role": "assistant", "content": "completed response"},
            ]

            import threading

            called = threading.Event()
            with patch("agent.title_generator.auto_title_session") as mock_auto:
                mock_auto.side_effect = lambda *a, **k: called.set()
                maybe_auto_title(
                    db,
                    "sess-burst",
                    "explain the screenshots",
                    "completed response",
                    history,
                )
                assert called.wait(timeout=10), "burst fallback auto-title never ran"

            mock_auto.assert_called_once()

    def test_compaction_scaffolding_does_not_suppress_first_exchange(self):
            """Synthetic user-role compaction turns are not later human exchanges."""
            db = MagicMock()
            db.get_session_title.return_value = None
            from agent.context_compressor import SUMMARY_PREFIX

            history = [
                {"role": "user", "content": "build the archive"},
                {"role": "assistant", "content": ""},
                {"role": "tool", "content": "large result"},
                # In-place compaction restores the human anchor, then appends
                # user-role reference scaffolding to preserve the active task.
                {"role": "user", "content": "build the archive"},
                {"role": "user", "content": f"{SUMMARY_PREFIX}\ncompressed state"},
                {"role": "assistant", "content": "compressed summary"},
                {
                    "role": "user",
                    "content": (
                        "[Your active task list was preserved across context compression]"
                        "\n- [>] verify"
                    ),
                },
                {"role": "assistant", "content": "archive is live"},
            ]

            import threading

            callback_titles = []
            callback_called = threading.Event()

            def title_callback(title):
                callback_titles.append(title)
                callback_called.set()

            db.get_conversation_root.return_value = "sess-1"
            db.set_auto_title_if_empty.return_value = True
            with patch(
                "agent.title_generator.generate_title",
                return_value="I Ching Archive",
            ):
                maybe_auto_title(
                    db,
                    "sess-1",
                    "build the archive",
                    "archive is live",
                    history,
                    title_callback=title_callback,
                )
                assert callback_called.wait(timeout=10), "title callback never ran"

            assert callback_titles == ["I Ching Archive"]
            db.set_auto_title_if_empty.assert_called_once_with(
                "sess-1",
                "I Ching Archive",
            )


class TestManagedTitleCollisionRecovery:
    def test_retries_numbered_fallback_after_a_concurrent_collision(self):
            from agent.title_generator import _persist_session_title

            db = MagicMock()
            db.set_auto_title_if_empty.side_effect = [
                ValueError("base in use"),
                ValueError("#2 won by another worker"),
                True,
            ]
            db.get_next_title_in_lineage.side_effect = [
                "ProjectAtlas #2",
                "ProjectAtlas #3",
            ]

            assert (
                _persist_session_title(db, "fresh", "ProjectAtlas")
                == "ProjectAtlas #3"
            )
            assert db.set_auto_title_if_empty.call_args_list[-1][0] == (
                "fresh",
                "ProjectAtlas #3",
            )

    def test_regenerates_a_distinctive_title_before_numbering(self, tmp_path):
            db = SessionDB(tmp_path / "state.db")
            db.create_session(session_id="existing", source="cli")
            db.create_session(session_id="fresh", source="telegram")
            db.set_session_title("existing", "Hermes")

            with patch(
                "agent.title_generator.generate_title",
                side_effect=["Hermes", "Amazon Agent"],
            ) as generate:
                seen = []
                auto_title_session(
                    db,
                    "fresh",
                    "Should this Amazon operator migrate from Claude to Hermes?",
                    "Let's compare the systems.",
                    title_callback=seen.append,
                )

            assert db.get_session_title("fresh") == "Amazon Agent"
            assert generate.call_count == 2
            assert generate.call_args_list[1].kwargs["avoid_titles"] == ["Hermes"]
            assert seen == ["Amazon Agent"]
