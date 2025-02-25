import fl_pipeline

def main():

    averaging_type = 'local' 

    if averaging_type == 'local':

        FL_weightedlocalgrads() #averaging the local gradients of silos
    else:

        FL_weightedlocalglobalgrads() # Averaging the local gradients and global gradients ----> averaging the local gradients of silos