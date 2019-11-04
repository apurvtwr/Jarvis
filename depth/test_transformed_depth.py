import unittest
import torch
import numpy as np

from transformed_depth_map import DepthMap, Rotation, Intrinsics

class TestIntrinsics(unittest.TestCase) :
	def test_intrinsics(self) :
		intrinsics = Intrinsics(torch.FloatTensor([1, 2, 3, 4, 5]).reshape(1, 5))
		expected = torch.FloatTensor([
			[1, 5, 3],
			[0, 2, 4],
			[0, 0, 1]
			])
		self.assertEqual(torch.sum((expected!=intrinsics.value[0]).float()),0, "Intrinsics equality must match")


class TestRotation(unittest.TestCase) :
	def test_identity(self) :
		rotation = Rotation(torch.FloatTensor([0,0,0]).reshape(1, 3))
		self.assertEqual(torch.Size([1, 3, 3]), rotation.value.shape, "Rotation must be [1, 3, 3] tensor")
		self.assertEqual(torch.sum((torch.eye(3)!=rotation.value[0]).float()),0, "Rotation must be identity")

	def test_rx(self) :
		theta = np.pi/4.
		rotation = Rotation(torch.FloatTensor([theta,0,0]).reshape(1, 3))
		rx = np.array([
			[1, 0, 0],
			[0, np.cos(theta), -np.sin(theta)],
			[0, np.sin(theta), np.cos(theta)]
			])
		self.assertEqual(torch.sum((torch.FloatTensor(rx)!=rotation.value[0]).float()),0, "Rx equality must match")

	def test_ry(self) :
		theta = np.pi/4.
		rotation = Rotation(torch.FloatTensor([0, theta,0]).reshape(1, 3))
		ry = np.array([
			[np.cos(theta), 0, np.sin(theta)],
			[0, 1, 0],
			[-np.sin(theta), 0, np.cos(theta)]
			])
		self.assertEqual(torch.sum((torch.FloatTensor(ry)!=rotation.value[0]).float()),0, "Ry equality must match")

	def test_rz(self) :
		theta = np.pi/4.
		rotation = Rotation(torch.FloatTensor([0, 0, theta]).reshape(1, 3))
		rz = np.array([
			[np.cos(theta), -np.sin(theta), 0],
			[np.sin(theta), np.cos(theta), 0],
			[0, 0, 1]
			])
		self.assertEqual(torch.sum((torch.FloatTensor(rz)!=rotation.value[0]).float()),0, "Rz equality must match")

def main():
	unittest.main()


if __name__ == '__main__':
	main()