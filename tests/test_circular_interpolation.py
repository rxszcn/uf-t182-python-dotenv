import io
import os
import pathlib
import subprocess
import sys
from typing import Dict, Optional

import pytest

import dotenv
from dotenv.main import CircularInterpolationError, resolve_variables


def values(string: str) -> Dict[str, Optional[str]]:
    return dict(dotenv.dotenv_values(stream=io.StringIO(string)))


# Keys that only reference other keys, and keys that only reference
# themselves, must keep the behaviour they had before cycle detection: file
# order decides what a forward reference sees, a self reference falls back to
# the environment / default / empty string.
@pytest.mark.parametrize(
    "string,expected",
    [
        # Backward reference: the referenced key is defined earlier.
        ("B=c\nA=${B}", {"B": "c", "A": "c"}),
        # Forward reference: the referenced key is defined later, so it is not
        # visible yet and resolution falls back to the empty string.
        ("A=${B}\nB=c", {"A": "", "B": "c"}),
        # Forward reference with a default.
        ("A=${B:-d}\nB=c", {"A": "d", "B": "c"}),
        # A chain, all backward.
        ("C=c\nB=${C}\nA=${B}", {"C": "c", "B": "c", "A": "c"}),
        # A chain that is only partially resolvable, as before.
        ("A=${B}\nB=${C}\nC=c", {"A": "", "B": "", "C": "c"}),
        # Self reference with no environment value and no default.
        ("A=${A}", {"A": ""}),
        # Self reference with a default.
        ("A=${A:-d}", {"A": "d"}),
        # Self reference mixed with surrounding text and another key.
        ("B=c\nA=x${A}y${B}", {"B": "c", "A": "xyc"}),
        # Redefining a key: the last definition is the one that resolves.
        ("A=b\nA=c\nD=${A}", {"A": "c", "D": "c"}),
        (
            "A=b\nC=${A}\nD=e\nC=${D}",
            {"A": "b", "C": "e", "D": "e"},
        ),
    ],
)
def test_non_circular_references_keep_existing_behaviour(string, expected):
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.delenv("A", raising=False)
        monkeypatch.delenv("B", raising=False)
        monkeypatch.delenv("C", raising=False)
        monkeypatch.delenv("D", raising=False)
        assert values(string) == expected


def test_self_reference_uses_environment():
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("T182_SELF", "from-environment")
        assert values("T182_SELF=${T182_SELF}") == {"T182_SELF": "from-environment"}
        assert values("T182_SELF=${T182_SELF:-d}") == {"T182_SELF": "from-environment"}


def test_self_reference_in_a_non_circular_file():
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("T182_SELF", "from-environment")
        assert values(
            "T182_SELF=${T182_SELF}\nT182_OTHER=${T182_BASE}\nT182_BASE=x"
        ) == {
            "T182_SELF": "from-environment",
            # Forward references are untouched: this key is still resolved
            # before T182_BASE is defined.
            "T182_OTHER": "",
            "T182_BASE": "x",
        }


@pytest.mark.parametrize(
    "string,expected_cycle",
    [
        ("A=${B}\nB=${A}", ("A", "B", "A")),
        # Written in the other order: the cycle starts at the first key.
        ("B=${A}\nA=${B}", ("B", "A", "B")),
        # A three-key cycle.
        ("A=${B}\nB=${C}\nC=${A}", ("A", "B", "C", "A")),
        # The cycle is reachable through a key that is not on it.
        ("X=${A}\nA=${B}\nB=${A}", ("A", "B", "A")),
        # Defaults do not break a two-key cycle: the result would still depend
        # on which key happens to be resolved first.
        ("A=${B:-d}\nB=${A}", ("A", "B", "A")),
        # A key that references itself and another key is on a real cycle.
        ("A=${A}${B}\nB=${A}", ("A", "B", "A")),
    ],
)
def test_circular_reference_raises(string, expected_cycle):
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.delenv("A", raising=False)
        monkeypatch.delenv("B", raising=False)
        monkeypatch.delenv("C", raising=False)
        monkeypatch.delenv("X", raising=False)

        with pytest.raises(CircularInterpolationError) as exc_info:
            values(string)

    assert exc_info.value.cycles == [expected_cycle]
    assert "circular variable reference detected" in str(exc_info.value)
    assert " -> ".join(expected_cycle) in str(exc_info.value)


def test_circular_reference_reports_every_cycle():
    string = "A=${B}\nB=${A}\nC=${D}\nD=${C}\nE=${A}"

    with pytest.raises(CircularInterpolationError) as exc_info:
        values(string)

    assert exc_info.value.cycles == [("A", "B", "A"), ("C", "D", "C")]


def test_cycle_through_a_default_is_reported():
    # The default is written syntactically, but which value wins still
    # depends on file order, so this is a real cycle.
    with pytest.raises(CircularInterpolationError):
        values("A=${B}\nB=${A:-d}")


def test_reported_cycles_are_not_empty_strings():
    with pytest.raises(CircularInterpolationError):
        values("T182_CYCLE_A=${T182_CYCLE_B}\nT182_CYCLE_B=${T182_CYCLE_A}")


def test_circular_reference_with_override_disabled():
    # The structure of the file is circular regardless of the environment
    # precedence, so the signal does not depend on `override`.
    with pytest.raises(CircularInterpolationError):
        resolve_variables(
            [("T182_OVR_A", "${T182_OVR_B}"), ("T182_OVR_B", "${T182_OVR_A}")],
            override=False,
        )


# A file can mix a legal self reference with a real cycle. Only the keys on
# the real cycle are a defect; the self-referencing key keeps falling back to
# the environment, as it always did.
@pytest.mark.parametrize(
    "string,expected_cycle",
    [
        # Self reference first, then a two-key cycle.
        ("S=${S:-x}\nA=${B}\nB=${A}", ("A", "B", "A")),
        # Cycle first, self reference later.
        ("A=${B}\nB=${A}\nS=${S:-x}", ("A", "B", "A")),
        # Self-referencing key between the two keys of the cycle.
        ("A=${B}\nS=${S:-x}\nB=${A}", ("A", "B", "A")),
        # Self reference and cycle share a key: A is on the cycle because of
        # its reference to B; the reference to itself is still legal.
        ("A=${A}${B}\nB=${A}", ("A", "B", "A")),
        # Two self references, one cycle.
        ("A=${A}\nB=${B:-y}\nC=${D}\nD=${C}", ("C", "D", "C")),
        # A self reference in a key that also feeds the cycle.
        ("S=${S:-x}\nA=${B}${S}\nB=${A}", ("A", "B", "A")),
    ],
)
def test_self_reference_mixed_with_cycle(string, expected_cycle):
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name in ("S", "A", "B", "C", "D"):
            monkeypatch.delenv(name, raising=False)

        with pytest.raises(CircularInterpolationError) as exc_info:
            values(string)

    assert exc_info.value.cycles == [expected_cycle]


def test_self_reference_mixed_with_cycle_reports_only_the_cycle():
    # The self references resolve normally; the error names only the keys
    # whose values would have depended on each other.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("S", "from-environment")
        with pytest.raises(CircularInterpolationError) as exc_info:
            values("S=${S}\nA=${B}\nB=${A}\nT=${T}")

    assert exc_info.value.cycles == [("A", "B", "A")]
    assert "S" not in str(exc_info.value)
    assert "T" not in str(exc_info.value)


def test_redefinition_breaking_a_cycle_is_not_reported():
    # The effective definition of A is the literal on the last line, so the
    # delivered value of A does not depend on B. B referenced A before that
    # literal was defined, so it gets the ordinary forward-reference fallback
    # (empty), not a circular value.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.delenv("A", raising=False)
        monkeypatch.delenv("B", raising=False)
        assert values("A=${B}\nB=${A}\nA=c") == {"A": "c", "B": ""}


def test_redefinition_keeping_a_cycle_is_reported():
    # Redefining B without removing the mutual reference keeps the cycle.
    with pytest.raises(CircularInterpolationError):
        values("A=${B}\nB=${A}\nB=${A:-d}")


def test_redefinition_replacing_the_reference_with_a_self_reference():
    # The final definition of A only refers to itself, so there is no mutual
    # reference and the ordinary self-reference fallback applies.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.delenv("A", raising=False)
        assert values("A=x\nA=${A}y") == {"A": "xy"}


def test_cycle_detection_ignores_environment_values():
    # Even when the environment could supply both keys, the file's final
    # definitions still reference each other, and the result would depend on
    # file order. A structural cycle is reported deterministically.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("T182_ENV_A", "ea")
        monkeypatch.setenv("T182_ENV_B", "eb")

        with pytest.raises(CircularInterpolationError):
            values("T182_ENV_A=${T182_ENV_B}\nT182_ENV_B=${T182_ENV_A}")


def test_resolve_variables_accepts_an_iterator():
    def entries():
        yield "A", "${B}"
        yield "B", "${A}"

    with pytest.raises(CircularInterpolationError):
        resolve_variables(entries(), override=True)


@pytest.mark.skipif(
    sys.platform == "win32", reason="This test assumes case-sensitive variable names"
)
def test_circular_reference_names_are_case_sensitive():
    # `a` and `A` are different keys in the file, so there is no self
    # reference and no cycle.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.delenv("a", raising=False)
        monkeypatch.delenv("A", raising=False)
        assert values("a=${A}\nA=x") == {"a": "", "A": "x"}


def test_load_dotenv_does_not_partially_update_environ(dotenv_path):
    dotenv_path.write_text("T182_LOAD_A=${T182_LOAD_B}\nT182_LOAD_B=${T182_LOAD_A}")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.delenv("T182_LOAD_A", raising=False)
        monkeypatch.delenv("T182_LOAD_B", raising=False)

        with pytest.raises(CircularInterpolationError):
            dotenv.load_dotenv(dotenv_path)

        assert "T182_LOAD_A" not in os.environ
        assert "T182_LOAD_B" not in os.environ


def test_get_key_raises_for_a_cycle(dotenv_path):
    dotenv_path.write_text("T182_GET_A=${T182_GET_B}\nT182_GET_B=${T182_GET_A}")

    with pytest.raises(CircularInterpolationError):
        dotenv.get_key(dotenv_path, "T182_GET_A")


def test_interpolation_disabled_keeps_references_literal(dotenv_path):
    # Without interpolation `${...}` is plain text, so there is nothing to
    # resolve and no cycle to report.
    dotenv_path.write_text("A=${B}\nB=${A}")

    assert dotenv.dotenv_values(dotenv_path, interpolate=False) == {
        "A": "${B}",
        "B": "${A}",
    }


@pytest.mark.skipif(
    sys.platform == "win32", reason="This test assumes case-sensitive variable names"
)
def test_long_reference_chain_does_not_hit_the_recursion_limit():
    # 5000 keys in a chain, each referencing the next one; the last points
    # back at the first, forming one large cycle.
    count = 5000
    lines = [f"K{i}=${{K{i + 1}}}" for i in range(count - 1)]
    lines.append(f"K{count - 1}=${{K0}}")

    with pytest.raises(CircularInterpolationError) as exc_info:
        values("\n".join(lines))

    (cycle,) = exc_info.value.cycles
    assert cycle[0] == cycle[-1] == "K0"
    assert len(set(cycle)) == count
    # The message is abbreviated rather than dumping all 5000 keys.
    message = str(exc_info.value)
    assert "5000 keys" in message
    assert len(message) < 500


def _run_module(args, cwd, env):
    return subprocess.run(
        [sys.executable, "-m", "dotenv", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
    )


def _module_env() -> Dict[str, str]:
    env = dict(os.environ)
    # Make the subprocess import the test's own dotenv package instead of
    # whatever happens to be installed in site-packages.
    env["PYTHONPATH"] = str(pathlib.Path(dotenv.__file__).resolve().parent.parent)
    return env


@pytest.mark.parametrize("command", [["list"], ["get", "T182_CLI_A"]])
def test_cli_reports_cycle_and_exits_2(tmp_path, command):
    (tmp_path / ".env").write_text("T182_CLI_A=${T182_CLI_B}\nT182_CLI_B=${T182_CLI_A}")

    result = _run_module([*command], cwd=tmp_path, env=_module_env())

    assert result.returncode == 2, result.stderr
    assert "circular variable reference detected" in result.stderr
    assert "T182_CLI_A -> T182_CLI_B -> T182_CLI_A" in result.stderr


def test_cli_run_does_not_start_the_command_for_a_cycle(tmp_path):
    (tmp_path / ".env").write_text("T182_CLI_A=${T182_CLI_B}\nT182_CLI_B=${T182_CLI_A}")

    result = _run_module(
        ["run", sys.executable, "-c", "print('started')"],
        cwd=tmp_path,
        env=_module_env(),
    )

    assert result.returncode == 2, result.stderr
    assert result.stdout == ""
    assert "circular variable reference detected" in result.stderr


@pytest.mark.parametrize(
    "string,args,expected",
    [
        (
            "T182_CLI_SELF=${T182_CLI_SELF:-fallback}",
            ["run", "printenv", "T182_CLI_SELF"],
            "fallback\n",
        ),
        (
            "T182_CLI_BASE=x\nT182_CLI_DERIVED=${T182_CLI_BASE}/suffix",
            ["run", "printenv", "T182_CLI_DERIVED"],
            "x/suffix\n",
        ),
    ],
)
def test_cli_non_circular_files_still_work(tmp_path, string, args, expected):
    (tmp_path / ".env").write_text(string)
    env = _module_env()
    env.pop("T182_CLI_SELF", None)
    env.pop("T182_CLI_DERIVED", None)

    result = _run_module(args, cwd=tmp_path, env=env)

    assert result.returncode == 0, result.stderr
    assert result.stdout == expected
