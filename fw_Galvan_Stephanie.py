# Author: Stephanie Galvan
# Class: Parallel Computing
# Instructor: David Pruitt

from mpi4py import MPI
import math
import timeit

def floyd_warshall(path='fwTest.txt'):
        # Get the world communicator
        comm = MPI.COMM_WORLD

        # get our rank (process #)
        rank = comm.Get_rank()

        # get the size of the communicator in # processes
        size = comm.Get_size()

        # Get matrix
        matrix = getMatrix(path)

        # Print original matrix
        if(rank == 0):
            print('Original (subarray) Matrix')
            printSubarray(matrix)
            print('')

        matrix_length = len(matrix)

        # Formulas to use
        row_per_threads= matrix_length // size

        threads_per_row = size / matrix_length

        row_start = math.trunc(row_per_threads*rank)

        row_end = math.trunc(row_per_threads*(rank + 1))

        # Calculation part of the Floyd-Warshall Algorithm
        for k in range(len(matrix)):
            row_owner = int(threads_per_row * k)
            matrix[k] = comm.bcast(matrix[k], root=row_owner)

            for x in range(row_start, row_end):
                for y in range(len(matrix)):
                    matrix[x][y] = min(int(matrix[x][y]), (int(matrix[x][k]) + int(matrix[k][y])))

        # Putting together the matrix
        if comm.Get_rank() == 0:
            for k in range(row_end, len(matrix)):
                row_owner = int(threads_per_row * k)
                matrix[k] = comm.recv(source=row_owner, tag=k)
            print('Resultant (subarray) Matrix')
            printSubarray(matrix)
            print('')
        else:
            for k in range(row_start, row_end):
                comm.send(matrix[k], dest=0, tag=k)

def getMatrix(path):
    """
    Reads file containing a n*n matrix and returns a matrix
    """
    opened_path = open(path, 'r')
    file = opened_path.readlines()
    for i in range(len(file)):
        # Split to have rows at the end
        file[i] = file[i].split()
    return file

def printSubarray(matrix, size=10):
    """
    Prints the upper left subarray of dimensions size x size of
    the matrix
    """
    matrix_length = len(matrix)
    if(matrix_length < 10):
        size = matrix_length

    for row in range(size):
        for col in range(size):
            print(f'{matrix[row][col]} ', end='')
        print('')

def main():
    comm = MPI.COMM_WORLD
    print('\nFloyd Warshall Algorithm\n', end='')
    print('Thread: ', (comm.Get_rank() + 1))
    start = timeit.default_timer()
    floyd_warshall()
    end = timeit.default_timer()
    print('runtime: ', (end - start))

main()
