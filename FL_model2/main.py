import fl_pipeline as pipe

def main():

    averaging_type = 'local' 

    if averaging_type == 'local':

        epoch, model_path, silo_loss = pipe.FL_weightedlocalgrads() #averaging the local gradients of silos
    else:

        epoch, model_path, silo_loss = pipe.FL_weightedlocalglobalgrads() # Averaging the local gradients and global gradients ----> averaging the local gradients of silos

    for silo in ['CamCAN','SALD','eNki']:
        print(f'silo name:{silo}',end='')
        test_path = f'/home/tanurima/germany/brain_age_parcels/{silo}/{silo}_test.csv'
        pipe.evaluate_model(test_path,model_path)
        print('\n')


main()