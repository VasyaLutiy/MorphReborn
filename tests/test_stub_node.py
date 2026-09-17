"""Tests for the stub inference node (``tests/stub_node.py``).

The node is test tooling, so it gets the same treatment as the code it serves:
if it lies about an answer, every integration test and the whole live demo lie
with it. Each case starts a real server on an ephemeral port in a thread and
talks to it over HTTP with ``urllib`` -- no mocks anywhere, because what is
under test IS the wire format that ``processors.llama_cpp_processor`` (through
the OpenAI SDK) expects. Nothing here sleeps; the suite costs the price of a few
localhost round trips.
"""

import glob
import json
import os
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

# pytest prepends this test's own directory to sys.path, so the node under test
# is importable as a plain module without making ``tests`` a package.
import stub_node

from cards.generations import response_to_files

ANSWERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "stub_answers.json")


class StubNodeTestCase(unittest.TestCase):
    """Base: starts a node per test and tears it down, whatever the test does."""

    def start_node(self, answers, log_dir=None, node_name="stub-a"):
        """A running node on a free port; returns that port."""
        server = stub_node.make_server(
            "127.0.0.1", 0, stub_node.AnswerBook(answers),
            node_name=node_name, log_dir=log_dir)
        # A short poll interval only so ``shutdown()`` returns promptly: the
        # default half-second, times one server per test, is most of the suite.
        thread = threading.Thread(target=server.serve_forever, args=(0.01,))
        thread.daemon = True
        thread.start()

        def stop():
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.addCleanup(stop)
        return server.server_address[1]

    def completion(self, port, prompt, stream):
        """One POST /v1/chat/completions; returns the answer text."""
        payload = {
            "model": "stub-model",
            "messages": [
                {"role": "system", "content": "You are a code generator."},
                {"role": "user", "content": prompt},
            ],
            "stream": stream,
        }
        request = urllib.request.Request(
            "http://127.0.0.1:%d/v1/chat/completions" % port,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8")
        return _assemble_stream(body) if stream else _assemble_completion(body)


def _assemble_completion(body):
    payload = json.loads(body)
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "stop", payload
    return choice["message"]["content"]


def _assemble_stream(body):
    """Reassemble an SSE body exactly the way the OpenAI SDK's consumer does.

    Also asserts the stream's shape: a terminating ``[DONE]``, and a final data
    frame carrying ``finish_reason: "stop"``. A stub that streams content but
    never finishes would hang a real client.
    """
    text = ""
    finish_reasons = []
    saw_done = False
    for line in body.split("\n"):
        if not line.startswith("data: "):
            continue
        data = line[len("data: "):].strip()
        if data == "[DONE]":
            saw_done = True
            continue
        assert not saw_done, "a frame arrived after [DONE]"
        frame = json.loads(data)
        choice = frame["choices"][0]
        text += choice["delta"].get("content") or ""
        if choice.get("finish_reason"):
            finish_reasons.append(choice["finish_reason"])
    assert saw_done, "the stream never sent [DONE]"
    assert finish_reasons == ["stop"], finish_reasons
    return text


class ChatCompletionTests(StubNodeTestCase):

    def test_streaming_request_reassembles_into_the_canned_answer(self):
        # Longer than one SSE chunk on purpose: reassembly is what is under test.
        answer = "FILE: a.py\n```python\n%s\n```\n" % ("x = 1\n" * 40)
        port = self.start_node({"two-files": answer})

        self.assertEqual(
            self.completion(port, "write it [[STUB:two-files]] now", stream=True),
            answer)

    def test_non_streaming_request_returns_the_same_text(self):
        answer = "```python\nVALUE = 999\n```\n"
        port = self.start_node({"one-file": answer})

        streamed = self.completion(port, "[[STUB:one-file]]", stream=True)
        whole = self.completion(port, "[[STUB:one-file]]", stream=False)

        self.assertEqual(whole, answer)
        self.assertEqual(whole, streamed)

    def test_marker_is_found_anywhere_in_any_message(self):
        port = self.start_node({"k": "answer"})

        prompt = "a long instruction\nwith the marker [[STUB:k]] buried in it\n"
        self.assertEqual(self.completion(port, prompt, stream=True), "answer")

    def test_attempt_keys_answer_differently_on_successive_calls(self):
        port = self.start_node({
            "cs-retry@1": "first, wrong",
            "cs-retry@2": "second, right",
            "other": "untouched",
        })

        first = self.completion(port, "[[STUB:cs-retry]]", stream=True)
        second = self.completion(port, "[[STUB:cs-retry]]", stream=False)
        # Another key's counter is its own.
        self.assertEqual(self.completion(port, "[[STUB:other]]", stream=True),
                         "untouched")
        third = self.completion(port, "[[STUB:cs-retry]]", stream=True)

        self.assertEqual(first, "first, wrong")
        self.assertEqual(second, "second, right")
        # Past the last declared attempt the last one sticks, rather than
        # collapsing into the misconfiguration answer.
        self.assertEqual(third, "second, right")


class MisconfigurationTests(StubNodeTestCase):
    """A scenario that names nothing must fail loudly, and must not kill the node."""

    def test_unknown_marker_key_returns_the_marked_answer(self):
        port = self.start_node({"known": "ok"})

        answer = self.completion(port, "[[STUB:nowhere]]", stream=True)

        self.assertIn("STUB NODE ERROR", answer)
        self.assertIn("nowhere", answer)

    def test_missing_marker_returns_the_marked_answer(self):
        port = self.start_node({"known": "ok"})

        answer = self.completion(port, "no marker at all here", stream=False)

        self.assertIn("STUB NODE ERROR", answer)
        self.assertIn("[[STUB:<key>]]", answer)

    def test_node_keeps_serving_after_a_misconfigured_request(self):
        port = self.start_node({"known": "ok"})

        self.completion(port, "[[STUB:nowhere]]", stream=True)
        self.completion(port, "no marker at all", stream=True)

        self.assertEqual(self.completion(port, "[[STUB:known]]", stream=True), "ok")

    def test_unknown_path_is_a_json_404(self):
        port = self.start_node({"known": "ok"})

        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(
                "http://127.0.0.1:%d/v1/embeddings" % port, timeout=10)

        self.assertEqual(raised.exception.code, 404)
        self.assertIn("error", json.loads(raised.exception.read().decode("utf-8")))


class ModelsEndpointTests(StubNodeTestCase):

    def test_models_lists_the_node_under_its_own_name(self):
        port = self.start_node({"known": "ok"}, node_name="stub-b")

        with urllib.request.urlopen(
                "http://127.0.0.1:%d/v1/models" % port, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(payload["object"], "list")
        self.assertEqual([model["id"] for model in payload["data"]], ["stub-b"])


class PromptLogTests(StubNodeTestCase):

    def test_log_dir_captures_the_full_prompt_of_every_request(self):
        log_dir = tempfile.mkdtemp(prefix="stub-node-log-")
        self.addCleanup(shutil.rmtree, log_dir, True)
        port = self.start_node({"cs-calc": "answer"}, log_dir=log_dir)

        self.completion(port, "context slice here [[STUB:cs-calc]]", stream=True)

        written = sorted(glob.glob(os.path.join(log_dir, "*.json")))
        self.assertEqual(len(written), 1)
        with open(written[0], "r", encoding="utf-8") as handle:
            record = json.load(handle)

        self.assertEqual(record["key"], "cs-calc")
        self.assertEqual(record["attempt"], 1)
        self.assertTrue(record["stream"])
        self.assertEqual([message["role"] for message in record["messages"]],
                         ["system", "user"])
        self.assertIn("context slice here", record["messages"][1]["content"])
        self.assertIn("cs-calc", os.path.basename(written[0]))

    def test_log_dir_is_created_when_missing(self):
        parent = tempfile.mkdtemp(prefix="stub-node-log-")
        self.addCleanup(shutil.rmtree, parent, True)
        log_dir = os.path.join(parent, "not", "there", "yet")

        self.start_node({"k": "v"}, log_dir=log_dir)

        self.assertTrue(os.path.isdir(log_dir))


class ShippedAnswersTests(unittest.TestCase):
    """The demo answers must be answers Morph can actually split into files.

    A typo in the ``FILE:`` form would only surface halfway through a live demo,
    as a card "corrupt response" -- so the shipped file is checked against the
    very function that splits a real answer.
    """

    def setUp(self):
        self.answers = stub_node.load_answers(ANSWERS_FILE)

    def test_documentation_keys_are_not_answers(self):
        self.assertNotIn("__about__", self.answers)

    def test_two_file_changeset_splits_into_a_module_and_its_test(self):
        targets = ["pkg_demo/calc.py", "tests/test_calc_demo.py"]
        files = response_to_files(self.answers["cs-calc"], targets)

        self.assertEqual(sorted(files), sorted(targets))
        self.assertIn("def add(a, b):", files["pkg_demo/calc.py"])
        self.assertIn("from pkg_demo.calc import add", files["tests/test_calc_demo.py"])
        for body in files.values():
            self.assertNotIn("```", body)

    def test_three_file_changeset_splits_into_its_three_files(self):
        targets = ["pkg_bad/one.py", "pkg_bad/two.py", "pkg_bad/three.py"]
        files = response_to_files(self.answers["cs-bad"], targets)

        self.assertEqual(sorted(files), sorted(targets))
        # The demo card asserts VALUE == 999, so this changeset must fail it.
        self.assertNotIn("999", files["pkg_bad/one.py"])

    def test_single_file_answer_is_one_bare_fenced_block(self):
        files = response_to_files(self.answers["cs-single"], ["pkg_demo/greet.py"])

        self.assertEqual(list(files), ["pkg_demo/greet.py"])
        self.assertIn("def greet(name):", files["pkg_demo/greet.py"])
        self.assertNotIn("FILE:", files["pkg_demo/greet.py"])

    def test_retry_attempts_go_from_failing_to_passing(self):
        book = stub_node.AnswerBook(self.answers)

        first, _, _ = book.resolve("[[STUB:cs-retry]]")
        second, _, _ = book.resolve("[[STUB:cs-retry]]")

        self.assertIn("VALUE = 0", first)
        self.assertIn("VALUE = 999", second)


class ServedByTheRealAnswersTests(StubNodeTestCase):
    """End to end on the shipped file: what a demo operator actually starts."""

    def test_shipped_answers_serve_over_the_wire(self):
        port = self.start_node(stub_node.load_answers(ANSWERS_FILE))

        answer = self.completion(port, "make a calculator [[STUB:cs-calc]]",
                                 stream=True)

        self.assertIn("FILE: pkg_demo/calc.py", answer)
        self.assertIn("FILE: tests/test_calc_demo.py", answer)


if __name__ == "__main__":
    unittest.main()
