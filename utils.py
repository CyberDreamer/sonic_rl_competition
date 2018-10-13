import numpy

def act_to_act(buttons):
    va = numpy.zeros(12)
    va[4] = buttons[0] # UP
    va[5] = buttons[1] # DOWN
    va[6] = buttons[2] # LEFT
    va[7] = buttons[3] # RIGHT
    return va