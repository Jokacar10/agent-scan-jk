"""Tests for --config-file: loading YAML, the precedence cascade, and
complete-replacement semantics for block/list arguments."""

import argparse

import pytest

from agent_scan.cli import (
    MissingIdentifierError,
    apply_config_file,
    control_servers_from_config,
    explicitly_provided_dests,
    load_config_file,
    parse_control_servers,
    setup_scan_parser,
)
from agent_scan.models import ControlServer


def _build_parser() -> argparse.ArgumentParser:
    """Build a parser mirroring the real ``scan`` subparser used in main().

    ``allow_abbrev=False`` matches main() so prefix abbreviations (e.g. ``--verb``)
    are rejected rather than silently expanded — this keeps
    ``explicitly_provided_dests`` (which matches full option strings) exact.
    """
    parser = argparse.ArgumentParser(allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command")
    scan_parser = subparsers.add_parser("scan", allow_abbrev=False)
    setup_scan_parser(scan_parser)
    return parser


def _parse(argv: list[str]) -> argparse.Namespace:
    """Parse ``argv`` (as sys.argv[1:]) and attach control_servers like main() does."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.control_servers = parse_control_servers(argv)
    return parser, args


def _write_yaml(tmp_path, text: str) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return str(path)


class TestLoadConfigFile:
    def test_loads_valid_mapping(self, tmp_path):
        path = _write_yaml(tmp_path, "server_timeout: 30\nverbose: true\n")
        assert load_config_file(path) == {"server_timeout": 30, "verbose": True}

    def test_empty_file_returns_empty_dict(self, tmp_path):
        path = _write_yaml(tmp_path, "")
        assert load_config_file(path) == {}

    def test_missing_file_exits_2(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            load_config_file(str(tmp_path / "does-not-exist.yaml"))
        assert exc.value.code == 2

    def test_invalid_yaml_exits_2(self, tmp_path):
        path = _write_yaml(tmp_path, "key: [unclosed\n")
        with pytest.raises(SystemExit) as exc:
            load_config_file(path)
        assert exc.value.code == 2

    def test_non_mapping_top_level_exits_2(self, tmp_path):
        path = _write_yaml(tmp_path, "- just\n- a\n- list\n")
        with pytest.raises(SystemExit) as exc:
            load_config_file(path)
        assert exc.value.code == 2


class TestExplicitlyProvidedDests:
    def test_detects_passed_flags_only(self):
        parser = _build_parser()
        provided = explicitly_provided_dests(parser, ["scan", "--server-timeout", "5", "--json"])
        assert "server_timeout" in provided
        assert "json" in provided
        assert "verbose" not in provided

    def test_detects_equals_form(self):
        parser = _build_parser()
        provided = explicitly_provided_dests(parser, ["scan", "--server-timeout=5"])
        assert "server_timeout" in provided

    def test_boolean_optional_both_spellings_map_to_same_dest(self):
        parser = _build_parser()
        assert "skills" in explicitly_provided_dests(parser, ["scan", "--no-skills"])
        assert "skills" in explicitly_provided_dests(parser, ["scan", "--skills"])


class TestAbbreviationDisabled:
    """main() sets allow_abbrev=False so prefix abbreviations are rejected, which
    keeps explicit-flag detection exact (an abbreviation would otherwise slip past
    the full-option-string match in explicitly_provided_dests)."""

    def test_abbreviated_flag_is_rejected(self):
        parser = _build_parser()
        # --verb is an unambiguous prefix of --verbose but must NOT be accepted.
        with pytest.raises(SystemExit):
            parser.parse_args(["scan", "--verb"])

    def test_full_flag_still_works(self):
        parser = _build_parser()
        assert parser.parse_args(["scan", "--verbose"]).verbose is True

    def test_config_merge_respects_explicit_full_flag(self, tmp_path):
        # With the full flag, the CLI value wins over YAML (regression guard for the
        # abbreviation gap: the merge must see verbose as explicitly provided).
        path = _write_yaml(tmp_path, "verbose: false\n")
        argv = ["scan", "--config-file", path, "--verbose"]
        parser, args = _parse(argv)
        apply_config_file(parser, args, argv)
        assert args.verbose is True


class TestControlServersFromConfig:
    def test_builds_with_headers_mapping(self):
        raw = [{"url": "https://s1.com", "identifier": "user1", "headers": {"Auth": "token1"}}]
        assert control_servers_from_config(raw) == [
            ControlServer(url="https://s1.com", headers={"Auth": "token1"}, identifier="user1")
        ]

    def test_builds_with_headers_list(self):
        raw = [{"url": "https://s1.com", "identifier": "user1", "headers": ["Auth: token1"]}]
        assert control_servers_from_config(raw) == [
            ControlServer(url="https://s1.com", headers={"Auth": " token1"}, identifier="user1")
        ]

    def test_missing_identifier_raises(self):
        with pytest.raises(MissingIdentifierError):
            control_servers_from_config([{"url": "https://s1.com"}])

    def test_non_list_exits_2(self):
        with pytest.raises(SystemExit) as exc:
            control_servers_from_config({"url": "https://s1.com"})
        assert exc.value.code == 2


class TestApplyConfigFileNoOp:
    def test_no_config_file_leaves_args_untouched(self):
        parser, args = _parse(["scan", "--server-timeout", "7"])
        before = vars(args).copy()
        apply_config_file(parser, args, ["scan", "--server-timeout", "7"])
        assert vars(args) == before

    def test_config_file_absent_keeps_defaults(self):
        parser, args = _parse(["scan"])
        apply_config_file(parser, args, ["scan"])
        assert args.server_timeout == 10  # code default preserved
        assert args.skills is True


class TestApplyConfigFileScalars:
    def test_yaml_value_fills_unpassed_flag(self, tmp_path):
        path = _write_yaml(tmp_path, "server_timeout: 30\nanalysis_url: https://yaml.example/api\n")
        argv = ["scan", "--config-file", path]
        parser, args = _parse(argv)
        apply_config_file(parser, args, argv)
        assert args.server_timeout == 30
        assert args.analysis_url == "https://yaml.example/api"

    def test_explicit_cli_flag_overrides_yaml(self, tmp_path):
        path = _write_yaml(tmp_path, "server_timeout: 30\n")
        argv = ["scan", "--config-file", path, "--server-timeout", "5"]
        parser, args = _parse(argv)
        apply_config_file(parser, args, argv)
        assert args.server_timeout == 5  # CLI wins over YAML

    def test_store_true_can_be_enabled_from_yaml(self, tmp_path):
        path = _write_yaml(tmp_path, "verbose: true\n")
        argv = ["scan", "--config-file", path]
        parser, args = _parse(argv)
        apply_config_file(parser, args, argv)
        assert args.verbose is True

    def test_boolean_optional_no_flag_overrides_yaml(self, tmp_path):
        # YAML enables skills, CLI --no-skills must win.
        path = _write_yaml(tmp_path, "skills: true\n")
        argv = ["scan", "--config-file", path, "--no-skills"]
        parser, args = _parse(argv)
        apply_config_file(parser, args, argv)
        assert args.skills is False

    def test_hyphenated_yaml_keys_accepted(self, tmp_path):
        path = _write_yaml(tmp_path, "server-timeout: 42\n")
        argv = ["scan", "--config-file", path]
        parser, args = _parse(argv)
        apply_config_file(parser, args, argv)
        assert args.server_timeout == 42

    def test_unknown_key_is_ignored(self, tmp_path, capsys):
        path = _write_yaml(tmp_path, "not_a_real_flag: 1\nserver_timeout: 15\n")
        argv = ["scan", "--config-file", path]
        parser, args = _parse(argv)
        apply_config_file(parser, args, argv)
        assert args.server_timeout == 15
        assert "not_a_real_flag" in capsys.readouterr().err


class TestApplyConfigFileControlServers:
    _YAML = (
        "control_servers:\n"
        "  - url: https://yaml-server.com\n"
        "    identifier: yaml-user\n"
        "    headers:\n"
        "      Auth: yaml-token\n"
    )

    def test_control_servers_loaded_from_yaml(self, tmp_path):
        path = _write_yaml(tmp_path, self._YAML)
        argv = ["scan", "--config-file", path]
        parser, args = _parse(argv)
        apply_config_file(parser, args, argv)
        assert args.control_servers == [
            ControlServer(url="https://yaml-server.com", headers={"Auth": "yaml-token"}, identifier="yaml-user")
        ]

    def test_cli_control_server_replaces_yaml_completely(self, tmp_path):
        # Passing any control-server block flag wipes the YAML array entirely.
        path = _write_yaml(tmp_path, self._YAML)
        argv = [
            "scan",
            "--config-file",
            path,
            "--control-server",
            "https://cli-server.com",
            "--control-identifier",
            "cli-user",
        ]
        parser, args = _parse(argv)
        apply_config_file(parser, args, argv)
        assert args.control_servers == [ControlServer(url="https://cli-server.com", headers={}, identifier="cli-user")]
        # The YAML server must be gone — no element-wise merge.
        assert all(cs.url != "https://yaml-server.com" for cs in args.control_servers)


class TestApplyConfigFileRepeatableHeaders:
    """--verification-H is a repeatable (append) array: same complete-replacement
    rule as control_servers — passing it on the CLI discards the YAML list."""

    _YAML = 'verification_H:\n  - "X-From-Yaml: a"\n  - "X-Second: b"\n'

    def test_verification_headers_loaded_from_yaml(self, tmp_path):
        path = _write_yaml(tmp_path, self._YAML)
        argv = ["scan", "--config-file", path]
        parser, args = _parse(argv)
        apply_config_file(parser, args, argv)
        assert args.verification_H == ["X-From-Yaml: a", "X-Second: b"]

    def test_cli_header_replaces_yaml_completely(self, tmp_path):
        # A single --verification-H on the CLI wipes the whole YAML array; no merge.
        path = _write_yaml(tmp_path, self._YAML)
        argv = ["scan", "--config-file", path, "--verification-H", "X-From-Cli: only"]
        parser, args = _parse(argv)
        apply_config_file(parser, args, argv)
        assert args.verification_H == ["X-From-Cli: only"]
        assert all("Yaml" not in h for h in args.verification_H)


class TestApplyConfigFileFiles:
    def test_files_loaded_from_yaml(self, tmp_path):
        path = _write_yaml(tmp_path, "files:\n  - /a/config.json\n  - /b/config.json\n")
        argv = ["scan", "--config-file", path]
        parser, args = _parse(argv)
        apply_config_file(parser, args, argv)
        assert args.files == ["/a/config.json", "/b/config.json"]

    def test_cli_positional_files_replace_yaml(self, tmp_path):
        path = _write_yaml(tmp_path, "files:\n  - /a/config.json\n")
        argv = ["scan", "/cli/config.json", "--config-file", path]
        parser, args = _parse(argv)
        apply_config_file(parser, args, argv)
        assert args.files == ["/cli/config.json"]
