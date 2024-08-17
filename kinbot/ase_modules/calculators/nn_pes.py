from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.core.common.relaxation.ase_utils import OCPCalculator

class Nn_surr(OCPCalculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='surrogate', atoms=None, tnsr=False, **kwargs):
        # well performing models:
        # EquiformerV2-lE4-lF100-S2EFS-OC22
        # EquiformerV2-31M-S2EF-OC20-All+MD
        # EquiformerV2-83M-S2EF-OC20-2M
        # eSCN-L6-M3-Lay20-S2EF-OC20-All+MD
        # EquiformerV2-153M-S2EF-OC20-All+MD
        checkpoint_path = model_name_to_local_file('EquiformerV2-153M-S2EF-OC20-All+MD', local_cache='/hpcwork/zo122003/BA/models')

        super().__init__(checkpoint_path=checkpoint_path, cpu=True)