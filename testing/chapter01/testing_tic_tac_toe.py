import unittest
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




