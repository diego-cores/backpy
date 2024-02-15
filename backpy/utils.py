"""
Utils.
----
Different useful functions for the operation of main code.
"""

def load_bar(size:int,step:int) -> None:
    """
    Loading bar.
    ----
    Print the loading bar.\n
    Parameters:
    --
    >>> size:int
    >>> step:int
    \n
    size: \n
    \tNumber of steps.\n
    step: \n
    \tstep.\n
    """
    per = str(int(step/size*100))
    load = '*'*int(46*step/size) + ' '*(46-int(46*step/size))

    first = load[:46//2-int(round(len(per)/2,0))]
    sec = load[46//2+int(len(per)-round(len(per)/2,0)):]

    print('\r['+first+per+'%%'+sec+']'+f'  {step} of {size} completed', end='')

def has_number_on_left(num:float) -> bool:
    """
    Has number on left.
    ----
    Returns true if there is a number other than 0 to the left of the '.'.
    """
    return str(num).lstrip('-0').partition('.')[0] != ''
