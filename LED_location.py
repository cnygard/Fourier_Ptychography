import numpy as np

def LED_location(xstart, ystart, arraysize):
    # LED lighting sequence spirals out
    # x+ y+ x- x- y- y- x+ x+ x+ y+ y+ y+ 
    # <<  <<  .  .  .  .   .
    # \/  .  16 15 14  13  .
    # \/  .   5  4  3  12  . 
    # \/  .   6  1  2  11  .
    # \/  .   7  8  9  10  .
    # \/  .   .  .  .  .   .
    # \/
    # Input: 
    #       xstart,ystart: absolute coordinate of initial LED
    #       arraysize: side length of lit LED array
    # Output:
    #       xlocation,ylocation: absolute coordinate of lit LEDs

    xorynode = np.zeros((1, 70)) # satisfy the 32*32 LED matrix
    xorynode[0, 0] = 1
    dif = 1
    dif_judge = 1
    for i in range(2, 71):
        xorynode[0, i - 1] = xorynode[0, i - 2] + dif
        if dif_judge < 2:
            dif_judge = dif_judge + 1
        else:
            dif = dif + 1
            dif_judge = 1
    
    # light LEDs with NO LED in the center
    xlocation = np.zeros((1, arraysize**2))
    ylocation = np.zeros((1, arraysize**2))
    xlocation[0, 0] = xstart
    ylocation[0, 0] = ystart
    xy_order = 2 # TODO: check this value? used for indexing so -1?
    for i in range(2, arraysize**2 + 1):
        if not i <= xorynode[0, xy_order]:
            xy_order = xy_order + 1
        if xy_order % 2 == 0:
            xlocation[0, i - 1] = xlocation[0, i - 2] + (-1)**((xy_order // 2) % 2 + 1)
            ylocation[0, i - 1] = ylocation[0, i - 2]
        elif xy_order % 2 == 1:
            xlocation[0, i - 1] = xlocation[0, i - 2]
            ylocation [0, i - 1] = ylocation[0, i - 2] + (-1)**(((xy_order - 1) // 2) % 2 + 1)
    
    return xlocation, ylocation
