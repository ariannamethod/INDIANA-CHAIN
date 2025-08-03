from code_exec import evaluate, run_python_snippet


def test_run_python_snippet_basic_execution():
    stdout, stderr, returncode = run_python_snippet("print('hello')")
    assert stdout.strip() == 'hello'
    assert stderr == ''
    assert returncode == 0


def test_evaluate_returns_reward_and_pass_status():
    code = "print(input()[::-1])"
    tests = [("abc\n", "cba"), ("123\n", "321")]
    result = evaluate(code, tests)
    assert result["passed"] is True
    assert result["reward"] == 1.0
    assert all(r["passed"] for r in result["results"])
