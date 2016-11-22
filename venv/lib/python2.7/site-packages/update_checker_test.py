#!/usr/bin/env python
import sys
import unittest
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO  # NOQA
from update_checker import UpdateChecker, update_check


class UpdateCheckerTest(unittest.TestCase):
    TRACKED_PACKAGE = 'praw'
    UNTRACKED_PACKAGE = 'requests'

    def test_check_check__bad_package(self):
        checker = UpdateChecker()
        checker.bypass_cache = True
        self.assertFalse(checker.check(self.UNTRACKED_PACKAGE, '0.0.1'))

    def test_checker_check__bad_url(self):
        checker = UpdateChecker('http://sdlkjsldfkjsdlkfj.com')
        checker.bypass_cache = True
        self.assertFalse(checker.check(self.TRACKED_PACKAGE, '0.0.1'))

    def test_checker_check__no_update_to_beta_version(self):
        checker = UpdateChecker()
        checker.bypass_cache = True
        self.assertFalse(checker.check(self.TRACKED_PACKAGE, '3.6'))

    def test_checker_check__update_to_beta_version_from_beta_version(self):
        checker = UpdateChecker()
        checker.bypass_cache = True
        self.assertTrue(checker.check(self.TRACKED_PACKAGE, '4.0.0b4'))

    def test_checker_check__update_to_rc_version_from_beta_version(self):
        checker = UpdateChecker()
        checker.bypass_cache = True
        self.assertTrue(checker.check(self.TRACKED_PACKAGE, '4.0.0b4'))

    def test_checker_check__successful(self):
        checker = UpdateChecker()
        checker.bypass_cache = True
        result = checker.check(self.TRACKED_PACKAGE, '1.0.0')
        self.assertTrue(result is not None)

    def test_update_check__failed(self):
        prev_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            update_check(self.UNTRACKED_PACKAGE, '0.0.1', bypass_cache=True)
        finally:
            result = sys.stdout
            sys.stdout = prev_stdout
        self.assertTrue(len(result.getvalue()) == 0)

    def test_update_check__successful(self):
        prev_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            update_check(self.TRACKED_PACKAGE, '0.0.1', bypass_cache=True)
        finally:
            result = sys.stdout
            sys.stdout = prev_stdout
        self.assertTrue(len(result.getvalue()) > 0)


if __name__ == '__main__':
    unittest.main()
