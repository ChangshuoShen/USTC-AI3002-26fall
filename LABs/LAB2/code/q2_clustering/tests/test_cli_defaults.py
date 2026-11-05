"""常规入口只暴露必需选项，辅助入口统一承接额外操作。"""
import argparse
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

import extras
import main


class ParsingComplete(Exception):
    pass


def parse_cli(module, arguments=()):
    original = argparse.ArgumentParser.parse_args
    parsed = None

    def capture(parser, args=None, namespace=None):
        nonlocal parsed
        parsed = original(parser, args, namespace)
        raise ParsingComplete

    with patch.object(sys, "argv", [module.__file__, *arguments]), \
         patch.object(argparse.ArgumentParser, "parse_args", autospec=True, side_effect=capture):
        try:
            module.main()
        except ParsingComplete:
            return parsed
    raise AssertionError("CLI did not parse arguments")


class CLIDefaultTests(unittest.TestCase):
    def test_checker_and_main_use_the_same_output(self):
        self.assertEqual(parse_cli(main).output, parse_cli(extras, ("check-output",)).output)
        self.assertEqual(parse_cli(main).output, PACKAGE / "outputs_final")

    def test_checker_and_extra_runs_use_the_formal_input(self):
        for command in ("check-output", "run", "evaluate"):
            arguments = (command, "--experiment", "fit") if command == "evaluate" else (command,)
            self.assertEqual(parse_cli(extras, arguments).input, main.DEFAULT_INPUT)

    def test_all_extra_commands_use_the_shared_submission(self):
        for command in ("run", "evaluate", "self-check"):
            arguments = [command]
            if command == "evaluate":
                arguments += ["--experiment", "fit"]
            with self.subTest(command=command):
                parsed = parse_cli(extras, arguments)
                self.assertEqual(parsed.implementation, PACKAGE / "submission.py")
                arguments += ["--implementation", "custom implementation/submission.py"]
                self.assertEqual(parse_cli(extras, arguments).implementation,
                                 Path("custom implementation/submission.py"))

    def test_explicit_output_with_spaces_is_preserved(self):
        output = "results with spaces/custom"
        self.assertEqual(parse_cli(main, ("--output", output)).output, Path(output))
        self.assertEqual(parse_cli(extras, ("check-output", "--output", output)).output, Path(output))

    def test_extra_fit_routes_to_main_without_evaluation(self):
        with patch.object(sys, "argv", ["extras.py", "run", "--stage", "fit", "--device", "cpu"]), \
             patch.object(main, "run", return_value=0) as run:
            with self.assertRaises(SystemExit) as stopped:
                extras.main()
        self.assertEqual(stopped.exception.code, 0)
        self.assertTrue(run.call_args.kwargs["fit_only"])


if __name__ == "__main__":
    unittest.main()
