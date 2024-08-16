import unittest
from cavsim2d.utils.shared_functions import *


class TestSharedFunctions(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_f2b_slashes(self):
        if os.name == 'nt':
            self.assertEqual(f2b_slashes('D:/User/directory'), 'D:\\User\\directory')
        else:
            self.assertEqual(f2b_slashes('D:\\User\\directory'), 'D:/User/directory')


if __name__ == '__main__':
    unittest.main()
