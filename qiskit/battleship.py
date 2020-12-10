#!/usr/bin/env python
# coding: utf-8

# <img src="https://raw.githubusercontent.com/Qiskit/qiskit-tutorials/master/images/qiskit-heading.png" alt="Note: In order for images to show up in this jupyter notebook you need to select File => Trusted Notebook" width="500 px" align="left">

# ## _*Quantum Battleships with partial NOT gates*_
# 
# The latest version of this notebook is available on https://github.com/qiskit/qiskit-tutorial.
# 
# ***
# ### Contributors
# James R. Wootton, University of Basel
# ***

# This program aims to act as an introduction to qubits, and to show how single-qubit operations can be used. Specifically, we'll use them to implement a game.
# 
# The game is based on the Japanese version of 'Battleships'. In this, each ship takes up only a single location. 
# 
# Each player will place three ships in the following five possible locations, which correspond to the five qubits of the ibmqx4 device.
# 
# <pre>
#                                                 4       0
#                                                 |\     /|
#                                                 | \   / |
#                                                 |  \ /  |
#                                                 |   2   |
#                                                 |  / \  |
#                                                 | /   \ |
#                                                 |/     \|
#                                                 3       1
# </pre>     
# 
# The players then fire bombs at each other's grids until one player loses all their ships.
# 
# The first ship placed by each player takes 1 bomb to destroy. The second ship takes 2, and the third takes 3.
# 
# The game mechanic is realized on a quantum computer by using a qubit for each ship, and using partial NOT gates (rotations around the Y axis) as the bombs. A full NOT is applied when the right number of bombs have hit a given ship, rotating the qubit/ship from 0 (undamaged) to 1 (destroyed).
# 
# Some details on implementation can be found in the Markdown cells. A full tutorial for how to program the game can be found at
# 
# https://medium.com/@decodoku/how-to-program-a-quantum-computer-982a9329ed02
# 
# If you are using the real device, here is a simple description of the game you can read while waiting for the jobs to finish.
# 
# https://medium.com/@decodoku/quantum-computation-in-84-short-lines-d9c7c74be0d0
# 
# If you just want to play, then select 'Restart & Run All' from the Kernel menu.
# <br>
# <br>

# First we import what we'll need to run set up and run the quantum program.

# In[1]:


from qiskit import IBMQ
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import execute


# Next we need to import *Qconfig*, which is found in the parent directory of this tutorial. Note that this will need to be set up with your API key if you haven't done so already.
# 
# After importing this information, it is used to register with the API. Then we are good to go!

# In[2]:


# Load the saved IBMQ account
IBMQ.load_account()


# Any quantum computation will really be a mixture of parts that run on a quantum device, and parts that run on a conventional computer. In this game, the latter consists of jobs such as getting inputs from players, and displaying the grid. The scripts for these are kept in a separate file, which we will import now.

# In[3]:


import sys
sys.path.append('game_engines')
from battleships_engine import *


# Now it's time for a title screen.

# In[4]:


title_screen()


# The player is now asked to choose whether to run on the real device (input *y* to do so).
# 
# The real device is awesome, of course, but you'll need to queue behind other people sampling its awesomeness. So for a faster experience, input *n* to simulate everything on your own (non-quantum) device.

# In[5]:


device = ask_for_device()


# The first step in the game is to get the players to set up their boards. Player 1 will be asked to give positions for three ships. Their inputs will be kept secret. Then the same for player 2.

# In[6]:


shipPos = ask_for_ships()


# The heart of every game is the main loop. For this game, each interation starts by asking players where on the opposing grid they want to bomb. The quantum computer then calculates the effects of the bombing, and the results are presented to the players. The game continues until all the ships of one player are destroyed.

# In[7]:


# the game variable will be set to False once the game is over
game = True

# the variable bombs[X][Y] will hold the number of times position Y has been bombed by player X+1
bomb = [ [0]*5 for _ in range(2)] # all values are initialized to zero

# set the number of samples used for statistics
shots = 1024

# the variable grid[player] will hold the results for the grid of each player
grid = [{},{}]

while (game):
    
    # ask both players where they want to bomb, and update the list of bombings so far
    bomb = ask_for_bombs( bomb )
    
    # now we create and run the quantum programs that implement this on the grid for each player
    qc = []
    for player in range(2):
        
        # now to set up the quantum program to simulate the grid for this player
        
        # set up registers and program
        q = QuantumRegister(5)
        c = ClassicalRegister(5)
        qc.append( QuantumCircuit(q, c) )
        
        # add the bombs (of the opposing player)
        for position in range(5):
            # add as many bombs as have been placed at this position
            for n in range( bomb[(player+1)%2][position] ):
                # the effectiveness of the bomb
                # (which means the quantum operation we apply)
                # depends on which ship it is
                for ship in [0,1,2]:
                    if ( position == shipPos[player][ship] ):
                        frac = 1/(ship+1)
                        # add this fraction of a NOT to the QASM
                        qc[player].u3(frac * math.pi, 0.0, 0.0, q[position])
                                        
        # Finally, measure them
        for position in range(5):
            qc[player].measure(q[position], c[position])

    # compile and run the quantum program
    job = execute(qc, backend=device, shots=shots)
    if not device.configuration().to_dict()['simulator']:
        print("\nWe've now submitted the job to the quantum computer to see what happens to the ships of each player\n(it might take a while).\n")
    else:
        print("\nWe've now submitted the job to the simulator to see what happens to the ships of each player.\n")
    # and extract data
    for player in range(2):
        grid[player] = job.result().get_counts(qc[player])
    
    game = display_grid ( grid, shipPos, shots )
        


# ## <br>
# <br>
# If you are reading this while running the game, you might be wondering where all the action has gone. Try clicking on the white space to the left of the output in the cell above to open it up.

# In[1]:


keywords = {'Topics': ['Games', 'NOT gates'], 'Commands': ['`u3`']}


# In[ ]:




