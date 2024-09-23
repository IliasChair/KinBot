from ocpmodels.common.relaxation.ase_utils import OCPCalculator

class Nn_surr(OCPCalculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='surrogate', atoms=None, tnsr=False, **kwargs):

        #checkpoint_path = model_name_to_local_file('EquiformerV2-lE4-lF100-S2EFS-OC22', local_cache='/hpcwork/zo122003/BA/models')
        super().__init__(checkpoint_path="/hpcwork/zo122003/BA/rundir_energy_forces/forces/DeNS_output/checkpoints/2024-09-02-02-10-40/best_checkpoint.pt", cpu=True, seed=1)