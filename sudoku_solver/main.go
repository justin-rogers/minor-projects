package main

import (
	"bufio"
	"log"
	"math"
	"os"
)

//sudokuData holds (attempted) solutions, i.e., the board state.
type sudokuData [9][9]byte

//parseSudoku reads the textual sudoku puzzles and returns a slice of
//sudokuData.
//Data from https://projecteuler.net/project/resources/p096_sudoku.txt
func parseSudoku() []sudokuData {
	file, err := os.Open("p096_sudoku.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	var D sudokuData
	sudokuJobs := []sudokuData{}
	//iterate on lines until EOF
	for count := 0; scanner.Scan(); count++ {
		line := scanner.Text()
		//every 10 lines is a separator
		i := count % 10
		if i == 0 {
			D = sudokuData{}
		} else { //populate the next block
			for j, c := range line {
				D[i-1][j] = byte(c - '0')
				//c - '0' subtracts ascii values, e.g., '3' - '0' = 3.
			}
		}
		if i == 9 {
			sudokuJobs = append(sudokuJobs, D)
			D = sudokuData{}
		}
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
	return sudokuJobs
}

//nextCoords traverses our array left-to-right, top-to-bottom.
//Roll back to 0,0 at the end.
func nextCoords(coords [2]byte) [2]byte {
	i, j := coords[0], coords[1]
	var x, y byte
	if j < 8 {
		x, y = i, j+1
	} else if i < 8 {
		x, y = i+1, 0
	} else { //end of array
		x, y = 0, 0
	}
	return [2]byte{x, y}
}

//getSubgrid computes the subgrid of a given point on the 9x9 grid.
//Each point on the 9x9 sudoku grid is contained in a unique 3x3 subgrid,
//and this function returns that subgrid as a slice.
//TODO: Memoize if efficiency is a problem.
func getSubgrid(coords [2]byte) [][2]byte {
	i, j := coords[0], coords[1]
	xShift := 3 * byte(math.Floor(float64(i/3)))
	yShift := 3 * byte(math.Floor(float64(j/3)))
	flatSubgrid := [][2]byte{}
	for x := byte(0); x < 3; x++ {
		for y := byte(0); y < 3; y++ {
			idx := [2]byte{xShift + x, yShift + y}
			flatSubgrid = append(flatSubgrid, idx)
		}
	}
	return flatSubgrid
}

//getCandidates returns a slice of the acceptable values at a given cell,
//and a bool that's true iff the cell was prepopulated.
//If the square is nonempty, there is only one acceptable value.
func (D *sudokuData) getCandidates(coords [2]byte) (candidates []byte, notEmpty bool) {
	i, j := coords[0], coords[1]
	if D[i][j] != 0 {
		candidates = []byte{D[i][j]}
		notEmpty = true
		return
	}
	notEmpty = false
	//Record all row, column, and 3x3 subgrid values, they aren't candidates.
	invalidNumbers := make(map[byte]bool)

	//rows
	for _, n := range D[i] {
		invalidNumbers[n] = true
	}
	//cols
	for k := 0; k < 9; k++ {
		invalidNumbers[D[k][j]] = true
	}
	//subgrid
	for _, idx := range getSubgrid(coords) {
		x, y := idx[0], idx[1]
		invalidNumbers[D[x][y]] = true
	}
	for v := byte(1); v < 10; v++ {
		_, bad := invalidNumbers[v]
		if !bad {
			candidates = append(candidates, v)
		}
	}
	return
}

//getExtension will use backtracking, starting at the given coordinates, to extend
//the given data to a solution, if one exists. It will return true when it finds
//a solution, or false if no such extension exists.
func (D *sudokuData) getExtension(coords [2]byte) bool {
	i, j := coords[0], coords[1]
	candidates, notEmpty := D.getCandidates(coords)

	//Prune dead branches of our search tree.
	if len(candidates) == 0 {
		return false
	}

	next := nextCoords(coords)
	//If we're at the final cell, we're done.
	if next == [2]byte{0, 0} {
		D[i][j] = candidates[0]
		return true
	}

	//If the cell is nonempty, i.e., it was one of the given squares, skip it.
	//This avoids erasing the given data by accident.
	if notEmpty {
		return D.getExtension(next)
	}

	//Core backtracking algorithm:
	//Fill in blank cells until we've rendered the puzzle unsolvable,
	//then erase the last cell and try the next candidate.
	solutionFound := false
	for _, v := range candidates {
		//Write a legal value.
		D[i][j] = v
		solutionFound = D.getExtension(next)
		if solutionFound {
			return true
		}
	}
	//No valid extension. Erase the last value we wrote.
	D[i][j] = 0
	return false
}

//solveForChksum will find a solution to a given sudokuData and return
//the first 3 digits as an int, as in the problem specification.
func (D *sudokuData) solveForChksum() int {
	success := D.getExtension([2]byte{0, 0})
	if !success {
		log.Fatal("No solution found", D)
	}
	//get the first 3 digits as an int
	return (int(D[0][0])*100 + int(D[0][1])*10 + int(D[0][2]))
}

func main() {
	puzzles := parseSudoku()
	chksum := 0
	for _, D := range puzzles {
		chksum += D.solveForChksum()
	}
	print(chksum)
}
