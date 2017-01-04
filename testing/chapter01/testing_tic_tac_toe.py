import unittest
import numpy as np
import numpy.testing as nptest
from chapter01 import tic_tac_toe as ttt


class TestingTicTacToe(unittest.TestCase):

    def assert_coordinate_bounds(self, coordinates):
        x, y = coordinates
        self.assertTrue(0 <= x <= 2)
        self.assertTrue(0 <= y <= 2)

    def test_index_to_board_coordinates(self):

        for i in range(0,9):
            coords = ttt.index_to_board_coordinates(i)
            self.assert_coordinate_bounds(coords)

    def test_index_to_board_coordinates_oob_lower(self):
        self.assertRaises(IndexError, ttt.index_to_board_coordinates, -1)

    def test_index_to_board_coordinates_oob_upper(self):
        self.assertRaises(IndexError, ttt.index_to_board_coordinates, 9)

    def test_string_to_state(self):
        string = '000000000'
        expected = np.ones((3,3)).astype(int) * 0
        state = ttt.string_to_state(string)
        nptest.assert_array_almost_equal(expected, state)

    def test_string_to_state_error(self):
        string = '0'
        self.assertRaises(ValueError, ttt.string_to_state, string)

    def test_string_to_state_char(self):
        string = 'WONT WORK'
        self.assertRaises(ValueError, ttt.string_to_state, string)




