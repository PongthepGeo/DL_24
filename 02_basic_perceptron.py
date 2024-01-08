#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import class_ as C
#-----------------------------------------------------------------------------------------#
import torch
#-----------------------------------------------------------------------------------------#

rock_types = {
    0: 'sedimentary',  # class_1
    1: 'metamorphic'   # class_2
}
number_of_nodes = 4

#-----------------------------------------------------------------------------------------#

perceptron = C.SimplePerceptron(input_size=number_of_nodes, rock_types=rock_types)
random_weights = torch.rand(number_of_nodes)
print("Random weights:", random_weights)
predicted_class = perceptron(random_weights)
print("Output rock type:", predicted_class)

#-----------------------------------------------------------------------------------------#