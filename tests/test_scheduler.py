import unittest

from scheduler import JobScheduler


class RotationSchedulingTests(unittest.TestCase):

    def setUp(self):
        self.scheduler = JobScheduler(["k80-a", "k80-b", "gpt4"])

    def test_first_three_rotation_jobs_dispatch_immediately_round_robin(self):
        j1 = self.scheduler.submit(None)
        j2 = self.scheduler.submit(None)
        j3 = self.scheduler.submit(None)

        self.assertEqual(j1.assigned, ["k80-a"])
        self.assertEqual(j2.assigned, ["k80-b"])
        self.assertEqual(j3.assigned, ["gpt4"])
        self.assertEqual(self.scheduler.busy_count(), 3)

    def test_fourth_and_fifth_rotation_jobs_queue_in_fifo_order(self):
        for _ in range(3):
            self.scheduler.submit(None)

        j4 = self.scheduler.submit(None)
        j5 = self.scheduler.submit(None)

        self.assertIsNone(j4.assigned)
        self.assertIsNone(j5.assigned)
        self.assertTrue(j4.was_queued)
        self.assertEqual(self.scheduler.queue_position(j4), 1)
        self.assertEqual(self.scheduler.queue_position(j5), 2)
        self.assertEqual(self.scheduler.queue_length(), 2)

    def test_queued_jobs_dequeue_in_fifo_order_when_slots_free(self):
        j1 = self.scheduler.submit(None)
        j2 = self.scheduler.submit(None)
        j3 = self.scheduler.submit(None)
        j4 = self.scheduler.submit(None)
        j5 = self.scheduler.submit(None)

        # k80-a (j1) finishes -> job 4 dequeues onto k80-a.
        self.scheduler.release(j1.assigned)
        self.assertEqual(j4.assigned, ["k80-a"])
        self.assertEqual(self.scheduler.queue_length(), 1)

        # gpt4 (j3) finishes -> job 5 dequeues onto gpt4.
        self.scheduler.release(j3.assigned)
        self.assertEqual(j5.assigned, ["gpt4"])
        self.assertEqual(self.scheduler.queue_length(), 0)

        # k80-b (j2) finishes -> queue is empty, nothing dispatched.
        self.scheduler.release(j2.assigned)
        self.assertEqual(self.scheduler.busy_count(), 2)  # j4 and j5 still running

    def test_cursor_is_unaffected_by_a_dequeue_dispatch(self):
        # Fill all three slots (cursor now wrapped back to k80-a == index 0).
        self.scheduler.submit(None)
        self.scheduler.submit(None)
        j3 = self.scheduler.submit(None)
        self.assertEqual(self.scheduler._cursor, 0)

        # Queue a fourth job and free gpt4 -> it dequeues onto gpt4, and the
        # cursor must stay put (a dequeue is not a fresh rotation pick).
        j4 = self.scheduler.submit(None)
        self.scheduler.release(j3.assigned)
        self.assertEqual(j4.assigned, ["gpt4"])
        self.assertEqual(self.scheduler._cursor, 0)

        # The next *fresh* rotation job still starts from the cursor (k80-a),
        # proving the cursor really did not silently advance above.
        self.scheduler.release(["k80-a"])
        j6 = self.scheduler.submit(None)
        self.assertEqual(j6.assigned, ["k80-a"])


class PinnedSchedulingTests(unittest.TestCase):

    def setUp(self):
        self.scheduler = JobScheduler(["k80-a", "k80-b", "gpt4"])

    def test_pinned_job_on_free_processor_starts_immediately(self):
        job = self.scheduler.submit(["gpt4"])
        self.assertEqual(job.assigned, ["gpt4"])

    def test_pinned_job_on_busy_processor_queues_for_that_processor_only(self):
        self.scheduler.submit(["gpt4"])  # occupies gpt4
        job = self.scheduler.submit(["gpt4"])

        self.assertIsNone(job.assigned)
        self.assertEqual(self.scheduler.queue_position(job), 1)
        # k80-a/k80-b are untouched and still free for rotation.
        rotation_job = self.scheduler.submit(None)
        self.assertEqual(rotation_job.assigned, ["k80-a"])

    def test_pinned_multi_processor_job_waits_for_all_pins_to_be_free(self):
        self.scheduler.submit(["k80-a"])       # busies k80-a
        self.scheduler.submit(["gpt4"])        # busies gpt4
        job = self.scheduler.submit(["k80-a", "gpt4"])
        self.assertIsNone(job.assigned)

        # Freeing only one of the two pins is not enough.
        self.scheduler.release(["k80-a"])
        self.assertIsNone(job.assigned)

        # Freeing the second pin dispatches it to both.
        self.scheduler.release(["gpt4"])
        self.assertEqual(job.assigned, ["k80-a", "gpt4"])

    def test_all_pins_fan_out_reserves_every_slot_and_blocks_rotation(self):
        job = self.scheduler.submit(["k80-a", "k80-b", "gpt4"])
        self.assertEqual(job.assigned, ["k80-a", "k80-b", "gpt4"])
        self.assertEqual(self.scheduler.busy_count(), 3)

        rotation_job = self.scheduler.submit(None)
        self.assertIsNone(rotation_job.assigned)
        self.assertEqual(self.scheduler.queue_position(rotation_job), 1)

    def test_rotation_does_not_steal_a_slot_a_pinned_job_is_waiting_on(self):
        self.scheduler.submit(["gpt4"])              # busies gpt4
        pinned = self.scheduler.submit(["gpt4"])      # queues for gpt4 specifically
        self.assertIsNone(pinned.assigned)

        # gpt4 frees: the waiting pinned job must claim it, not a rotation job
        # submitted afterwards.
        self.scheduler.release(["gpt4"])
        self.assertEqual(pinned.assigned, ["gpt4"])

    def test_two_multi_pin_jobs_resolve_in_arrival_order_without_deadlock(self):
        self.scheduler.submit(["k80-a"])  # busy k80-a
        self.scheduler.submit(["k80-b"])  # busy k80-b

        j1 = self.scheduler.submit(["k80-a", "k80-b"])
        j2 = self.scheduler.submit(["k80-b", "k80-a"])

        self.scheduler.release(["k80-a"])
        self.assertIsNone(j1.assigned)
        self.assertIsNone(j2.assigned)

        self.scheduler.release(["k80-b"])
        self.assertEqual(j1.assigned, ["k80-a", "k80-b"])
        self.assertIsNone(j2.assigned)

        self.scheduler.release(j1.assigned)
        self.assertEqual(j2.assigned, ["k80-b", "k80-a"])


class LaunchCallbackTests(unittest.TestCase):

    def setUp(self):
        self.scheduler = JobScheduler(["k80-a", "gpt4"])

    def test_launch_fires_immediately_when_already_dispatched(self):
        calls = []
        job = self.scheduler.submit(["k80-a"])
        self.scheduler.attach_launch(job, lambda: calls.append(job.id))
        self.assertEqual(calls, [job.id])

    def test_launch_is_deferred_until_the_job_is_actually_dispatched(self):
        calls = []
        self.scheduler.submit(["k80-a"])  # occupies the only pin target
        job = self.scheduler.submit(["k80-a"])
        self.scheduler.attach_launch(job, lambda: calls.append(job.id))
        self.assertEqual(calls, [])

        self.scheduler.release(["k80-a"])
        self.assertEqual(calls, [job.id])

    def test_launch_fires_at_most_once(self):
        calls = []
        job = self.scheduler.submit(["k80-a"])
        self.scheduler.attach_launch(job, lambda: calls.append(1))
        self.scheduler.attach_launch(job, lambda: calls.append(2))
        self.assertEqual(calls, [1])


if __name__ == "__main__":
    unittest.main()
