import pandas as pd
import numpy as np

def get_derivatives(df, kernel='sobel', padding='constant', values='relative'):
    """
    Computes the derivatives by convolving the "signal" with a kernel
    df = the dataframe to consider (this is a very limited usecase because it assumes the same column names all the time)
    kernel = the kernel type. so far i only support sobel cause its easy 😅
    padding = the padding mode for the "signal" can be anything that is supported in numpy.pad this variable just gets passed there
    values = the value type can be absolute or relative. if absoute it does nothing, if relative it assumes that t0 is not a normalized value and sets them all to 1
    """

    if values == 'relative':
        df.loc[:,"t0_volume"]=1
    elif values == 'absolute':
        pass
    else:
        raise ValueError(f"got unknown value mode {values}")
    if kernel == 'sobel':
        kernel = np.asarray([-1, 0, 1])
    else:
        raise ValueError(f'Got unkown kernel name {kernel}')
    
    derivatives = []
    for i, row in df.iterrows():
        row = np.asarray(row)
        row = row/row[0]
        der = np.zeros(6)
        row = np.pad(row, (1, 1), mode=padding)

        for j in range(2, len(row)-1):
            der[j-2] = kernel[0]*row[j+1] + kernel[1]*row[j] + kernel[2]*row[j-1]

        derivatives.append(der)
  
    ddf = pd.DataFrame(derivatives, columns=['t1_dr', 't2_dr','t3_dr','t4_dr','t5_dr','t6_dr'])
    print(ddf)
    return ddf