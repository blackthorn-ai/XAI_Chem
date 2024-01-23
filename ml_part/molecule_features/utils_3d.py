import math
from rdkit import Chem
from rdkit.Chem import rdchem, rdMolTransforms, rdForceFieldHelpers, rdPartialCharges
from rdkit.Chem import AllChem, Descriptors
from rdkit.Geometry import Point3D
from rdkit.Chem.rdchem import RWMol


carboxile = 'CC=O'
nitro_amine = 'CN'

submol1 = Chem.MolFromSmiles(carboxile)
submol2 = Chem.MolFromSmiles(nitro_amine)

class Molecule3DFeatures:
    def __init__(self, smiles, f_group, identificator) -> None:

        self.functional_group_to_smiles = {
            "CF3": "CC(F)(F)F", 
            "CH2F": "CCF", 
            "gem-CF2": "C(F)(F)", 
            "CHF2": "CC(F)(F)",
            "CHF": "CF",
            "non-F": ""
        }
        self.f_group = f_group
        self.identificator = identificator

        self.smiles = smiles
        self.mol = Molecule3DFeatures.prepare_molecule(smiles)
        self.min_energy_conf_index, self.min_energy = Molecule3DFeatures.find_conf_with_min_energy(self.mol)
        
        self.dipole_moment = self.calculate_dipole_moment()
        self.dihedral_angle_value = self.calculate_dihedral_angle()
        
        self.distance_between_atoms_in_cycle = self.calculate_distance_between_atoms_in_cycle()
        self.distance_between_atoms_in_f_group_centers = self.calculate_distance_between_atoms_in_f_group_centers()

        self.flat_angle_between_atoms_in_cycle_1, self.flat_angle_between_atoms_in_cycle_2 = self.calculate_flat_angle_between_atoms_in_cycle()
        self.flat_angle_between_atoms_in_f_group_center_1, self.flat_angle_between_atoms_in_f_group_center_2 = self.calculate_flat_angle_between_atoms_in_f_group_center()


    @staticmethod
    def prepare_molecule(smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        rdForceFieldHelpers.MMFFSanitizeMolecule(mol)
        
        num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        amount_of_confs = pow(3, num_rotatable_bonds + 3)
        AllChem.EmbedMultipleConfs(mol, numConfs=amount_of_confs, randomSeed=3407)

        rdPartialCharges.ComputeGasteigerCharges(mol)

        return mol

    
    @staticmethod
    def find_conf_with_min_energy(mol):
        optimization_result = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol)
        
        min_energy, min_energy_conf_index = pow(10,5), None
        for index, (status, energy) in enumerate(optimization_result):
            if energy < min_energy and status == 0:
                min_energy_conf_index = index
                min_energy = min(min_energy, energy)

        return min_energy_conf_index, min_energy


    @staticmethod
    def set_average_atoms_position(mol,
                                   atoms_idx: list(),
                                   conf_id:int):
        # returns mol and new atom id
        
        x, y, z = 0, 0, 0
        for atom_idx in atoms_idx:
            x += mol.GetConformer(conf_id).GetAtomPosition(atom_idx)[0]
            y += mol.GetConformer(conf_id).GetAtomPosition(atom_idx)[0]
            z += mol.GetConformer(conf_id).GetAtomPosition(atom_idx)[0]
        
        x /= len(atoms_idx)
        y /= len(atoms_idx)
        z /= len(atoms_idx)

        new_atom = rdchem.Atom(0)
        editable_molecule = RWMol(mol)
        idx = editable_molecule.AddAtom(new_atom)
        mol = editable_molecule.GetMol()

        conf = mol.GetConformer(conf_id)
        conf.SetAtomPosition(idx, Point3D(x,y,z))

        return mol, idx


    @staticmethod
    def dihedral_angle(mol, 
                       iAtomId:int, jAtomId:int, kAtomId:int, lAtomId:int, 
                       conf_id:int):
        
        conf = mol.GetConformer(conf_id)

        return rdMolTransforms.GetDihedralDeg(conf, iAtomId, jAtomId, kAtomId, lAtomId)


    def calculate_dihedral_angle(self):
        f_group_smiles = self.functional_group_to_smiles[self.f_group]

        carboxile_submol = Chem.MolFromSmiles('CC=O')
        nitro_amine_submol = Chem.MolFromSmiles('CN')

        carboxile_matches = self.mol.GetSubstructMatches(carboxile_submol)
        nitro_amine_matches = self.mol.GetSubstructMatches(nitro_amine_submol)

        X1, R1 = None, None
        if self.identificator == 'Carboxylic acid':
            if len(carboxile_matches) == 0:
                raise "Problem with carboxile acid"
            
            X1 = carboxile_matches[0][0]
            R1 = carboxile_matches[0][1]

        if "amine" in self.identificator.lower():
            if len(nitro_amine_matches) == 0:
                raise "Problem with amine"
            
            if "primary" in self.identificator.lower():
                X1 = nitro_amine_matches[0][0]
                R1 = nitro_amine_matches[0][1]
            elif "secondary" in self.identificator.lower():
                X1 = nitro_amine_matches[0][1]
                self.mol, R1 = Molecule3DFeatures.set_average_atoms_position(self.mol, [nitro_amine_matches[0][0], nitro_amine_matches[1][0]], self.min_energy_conf_index)

        X2, R2 = None, None
        f_group_submol = Chem.MolFromSmiles(f_group_smiles)
        f_group_matches = self.mol.GetSubstructMatches(f_group_submol)
        if self.f_group.upper() in ['CF3', 'CHF2', 'CH2F']:
            X2 = f_group_matches[0][0]
            R2 = f_group_matches[0][1]

        elif self.f_group == 'gem-CF2':
            X2 = f_group_matches[0][0]
            self.mol, R2 = Molecule3DFeatures.set_average_atoms_position(self.mol, [f_group_matches[0][1], f_group_matches[0][2]], self.min_energy_conf_index)

        elif self.f_group.upper() == 'CHF':
            if len(f_group_matches) == 1:
                X2 = f_group_matches[0][0]
                R2 = f_group_matches[0][1]
            elif len(f_group_matches) == 2:
                self.mol, X2 = Molecule3DFeatures.set_average_atoms_position(self.mol, [f_group_matches[0][0], f_group_matches[1][0]], self.min_energy_conf_index)
                self.mol, R2 = Molecule3DFeatures.set_average_atoms_position(self.mol, [f_group_matches[0][1], f_group_matches[1][1]], self.min_energy_conf_index)

        self.X1, self.X2, self.R1, self.R2 = X1, X2, R1, R2

        if len(set([X1, X2, R1, R2])) != 4:
            if len(set([X1, X2, R1, R2])) > 1:
                print("SAME x1 or x2 or r1 or r2", self.smiles)
            return None

        dihedral_angle_value = Molecule3DFeatures.dihedral_angle(self.mol, R2, X2, X1, R1, self.min_energy_conf_index) 
        print(f"X1: {X1}, R1: {R1}, X2: {X2}, R2: {R2}, smiles: {self.smiles}, identiicator: {self.identificator}, f_group: {self.f_group}, dihedral angle: {dihedral_angle_value}") 
        return dihedral_angle_value
    

    def calculate_distance_between_atoms_in_cycle(self):
        # returns r = |X1X2|

        if len(set([self.X1, self.X2, self.R1, self.R2])) != 4:
            return None

        X1_pos = self.mol.GetConformer(self.min_energy_conf_index).GetAtomPosition(self.X1)
        X2_pos = self.mol.GetConformer(self.min_energy_conf_index).GetAtomPosition(self.X2)

        r_vector = (X2_pos[0] - X1_pos[0], X2_pos[1] - X1_pos[1], X2_pos[2] - X1_pos[2])
        r_distance = math.sqrt(pow(2, r_vector[0]) + pow(2, r_vector[1]) + pow(2, r_vector[2]))

        return r_distance
    

    @staticmethod
    def flat_angle(mol, 
                   iAtomId:int, jAtomId:int, kAtomId:int, 
                   conf_id:int):
        
        conf = mol.GetConformer(conf_id)

        return rdMolTransforms.GetAngleDeg(conf, iAtomId, jAtomId, kAtomId)
    

    def calculate_distance_between_atoms_in_f_group_centers(self):
        # returns R = |R1R2|

        if len(set([self.X1, self.X2, self.R1, self.R2])) != 4:
            return None

        X1_pos = self.mol.GetConformer(self.min_energy_conf_index).GetAtomPosition(self.R1)
        X2_pos = self.mol.GetConformer(self.min_energy_conf_index).GetAtomPosition(self.R2)

        R_vector = (X2_pos[0] - X1_pos[0], X2_pos[1] - X1_pos[1], X2_pos[2] - X1_pos[2])
        R_distance = math.sqrt(pow(2, R_vector[0]) + pow(2, R_vector[1]) + pow(2, R_vector[2]))

        return R_distance


    def calculate_flat_angle_between_atoms_in_cycle(self):
        # returns two flat angles(in Deg) between 2 atoms in the cycle 
        # and 1 atom connected to the functional group:
        # 1) X1X2R2; 
        # 2) X2X1R1.
        if len(set([self.X1, self.X2, self.R1, self.R2])) != 4:
            return None, None

        flat_angle_between_atoms_in_cycle_1 = Molecule3DFeatures.flat_angle(self.mol, self.X1, self.X2, self.R2, self.min_energy_conf_index)
        flat_angle_between_atoms_in_cycle_2 = Molecule3DFeatures.flat_angle(self.mol, self.X2, self.X1, self.R1, self.min_energy_conf_index)

        return flat_angle_between_atoms_in_cycle_1, flat_angle_between_atoms_in_cycle_2


    def calculate_flat_angle_between_atoms_in_f_group_center(self):
        # returns two flat angles(in Deg) between 2 atoms in the cycle 
        # and 1 atom connected to the functional group:
        # 1) R2X2R1; 
        # 2) R1X1R2.
        if len(set([self.X1, self.X2, self.R1, self.R2])) != 4:
            return None, None

        flat_angle_between_atoms_in_f_group_center_1 = Molecule3DFeatures.flat_angle(self.mol, self.R2, self.X2, self.R1, self.min_energy_conf_index)
        flat_angle_between_atoms_in_f_group_center_2 = Molecule3DFeatures.flat_angle(self.mol, self.R1, self.X1, self.R2, self.min_energy_conf_index)

        return flat_angle_between_atoms_in_f_group_center_1, flat_angle_between_atoms_in_f_group_center_2


    def calculate_dipole_moment(self):

        charges = []
        coordinates = []
        for atom in self.mol.GetAtoms():
            pos = self.mol.GetConformer(self.min_energy_conf_index).GetAtomPosition(atom.GetIdx())
            charge = atom.GetDoubleProp("_GasteigerCharge")

            charges.append(charge)
            coordinates.append(pos)

        charges_multiply_coordinates = coordinates.copy()
        for charges_multiply_coordinate_index in range(len(charges_multiply_coordinates)):
            for coordinate in charges_multiply_coordinates[charges_multiply_coordinate_index]:
                coordinate *= charges[charges_multiply_coordinate_index]

        dipole_moment_vector = [0, 0, 0]
        for charges_multiply_coordinate_index in range(len(charges_multiply_coordinates)):
            dipole_moment_vector[0] += charges_multiply_coordinates[charges_multiply_coordinate_index][0]
            dipole_moment_vector[1] += charges_multiply_coordinates[charges_multiply_coordinate_index][1]
            dipole_moment_vector[2] += charges_multiply_coordinates[charges_multiply_coordinate_index][2]

        dipole_moment = math.sqrt(pow(dipole_moment_vector[0], 2) + pow(dipole_moment_vector[1], 2) + pow(dipole_moment_vector[2], 2))

        return dipole_moment
