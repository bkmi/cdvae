import torch
import torch.nn as nn
import torch.nn.functional as F

from cdvae.pl_modules.embeddings import MAX_ATOMIC_NUM
from cdvae.pl_modules.gemnet.gemnet import GemNetT
from cdvae.pl_modules.scn.scn import SphericalChannelNetwork
from cdvae.pl_modules.escn.escn import eSCN


def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


class GemNetTDecoder(nn.Module):
    """Decoder with GemNetT."""

    def __init__(
        self,
        hidden_dim=128,
        latent_dim=256,
        max_neighbors=20,
        radius=6.,
        scale_file=None,
        conditioning_dim: int = 0,
    ):
        super(GemNetTDecoder, self).__init__()
        self.cutoff = radius
        self.max_num_neighbors = max_neighbors
        self.conditioning_dim = conditioning_dim

        self.gemnet = GemNetT(
            num_targets=1,
            latent_dim=latent_dim,
            emb_size_atom=hidden_dim,
            emb_size_edge=hidden_dim,
            regress_forces=True,
            cutoff=self.cutoff,
            max_neighbors=self.max_num_neighbors,
            otf_graph=True,
            scale_file=scale_file,
            conditioning_dim=conditioning_dim,
        )
        self.fc_atom = nn.Linear(hidden_dim, MAX_ATOMIC_NUM)

    def forward(self, z, pred_frac_coords, pred_atom_types, num_atoms,
                lengths, angles, conditional=None):
        """
        args:
            z: (N_cryst, num_latent)
            pred_frac_coords: (N_atoms, 3)
            pred_atom_types: (N_atoms, ), need to use atomic number e.g. H = 1
            num_atoms: (N_cryst,)
            lengths: (N_cryst, 3)
            angles: (N_cryst, 3)
        returns:
            atom_frac_coords: (N_atoms, 3)
            atom_types: (N_atoms, MAX_ATOMIC_NUM)
        """
        # (num_atoms, hidden_dim) (num_crysts, 3)
        h, pred_cart_coord_diff = self.gemnet(
            z=z,
            frac_coords=pred_frac_coords,
            atom_types=pred_atom_types,
            num_atoms=num_atoms,
            lengths=lengths,
            angles=angles,
            edge_index=None,
            to_jimages=None,
            num_bonds=None,
            conditional=conditional,
        )
        pred_atom_types = self.fc_atom(h)
        return pred_cart_coord_diff, pred_atom_types


class SCNDecoder(nn.Module):
    """Decoder with SCN."""

    def __init__(
        self,
        max_neighbors=20,
        radius=8.0,
        max_num_elements=MAX_ATOMIC_NUM,
        num_interactions=8,
        lmax=6,
        mmax=1,
        num_resolutions=2,
        sphere_channels=128,
        sphere_channels_reduce=128,
        hidden_channels=256,
        num_taps=-1,
        use_grid=True,
        num_bands=1,
        num_sphere_samples=128,
        num_basis_functions=128,
        distance_function="gaussian",
        basis_width_scalar=1.0,
        distance_resolution=0.02,

        conditioning_dim: int = 0,
    ):
        super().__init__()
        self.cutoff = radius
        self.max_num_neighbors = max_neighbors
        self.conditioning_dim = conditioning_dim

        self.gemnet = SphericalChannelNetwork(
            num_atoms=0,  # not used
            bond_feat_dim=0,  # not used
            num_targets=1,  # not used
            use_pbc=True,
            regress_forces=True,
            otf_graph=True,
            max_num_neighbors=max_neighbors,
            cutoff=radius,
            max_num_elements=max_num_elements,
            num_interactions=num_interactions,
            lmax=lmax,
            mmax=mmax,
            num_resolutions=num_resolutions,
            sphere_channels=sphere_channels,
            sphere_channels_reduce=sphere_channels_reduce,
            hidden_channels=hidden_channels,
            num_taps=num_taps,
            use_grid=use_grid,
            num_bands=num_bands,
            num_sphere_samples=num_sphere_samples,
            num_basis_functions=num_basis_functions,
            distance_function=distance_function,
            basis_width_scalar=basis_width_scalar,
            distance_resolution=distance_resolution,
            show_timing_info=False,
            direct_forces=True,
        )
        self.fc_atom = nn.Linear(sphere_channels_reduce, MAX_ATOMIC_NUM)

    def forward(self, z, pred_frac_coords, pred_atom_types, num_atoms,
                lengths, angles, conditional=None):
        """
        args:
            z: (N_cryst, num_latent)
            pred_frac_coords: (N_atoms, 3)
            pred_atom_types: (N_atoms, ), need to use atomic number e.g. H = 1
            num_atoms: (N_cryst,)
            lengths: (N_cryst, 3)
            angles: (N_cryst, 3)
        returns:
            atom_frac_coords: (N_atoms, 3)
            atom_types: (N_atoms, MAX_ATOMIC_NUM)
        """
        # (num_atoms, hidden_dim) (num_crysts, 3)
        h, pred_cart_coord_diff = self.gemnet(
            z=z,
            frac_coords=pred_frac_coords,
            atom_types=pred_atom_types,
            num_atoms=num_atoms,
            lengths=lengths,
            angles=angles,
            edge_index=None,
            to_jimages=None,
            num_bonds=None,
            conditional=conditional,
        )
        pred_atom_types = self.fc_atom(h)
        return pred_cart_coord_diff, pred_atom_types


class ESCNDecoder(nn.Module):
    def __init__(
        self,
        max_neighbors=20,
        radius=8.0,
        max_num_elements=MAX_ATOMIC_NUM,

        use_pbc=True,
        regress_forces=True,
        regress_stress=False,
        outer_product_stress=True,
        decomposition_stress=False,
        edge_level=True,
        mixing_coordinates=False,
        extensive_energy=False,
        extensive_stress=False,
        otf_graph=False,
        num_layers=8,
        lmax_list=[6],
        mmax_list=[2],
        sphere_channels=128,
        hidden_channels=256,
        edge_channels=128,
        use_grid=True,
        num_sphere_samples=128,
        distance_function="gaussian",
        basis_width_scalar=1.0,
        distance_resolution=0.02,
        show_timing_info=True,
        enforce_max_neighbors_strictly=True,

        conditioning_dim: int = 0,
    ):
        
        super().__init__()
        self.cutoff = radius
        self.max_num_neighbors = max_neighbors
        self.conditioning_dim = conditioning_dim
        self.escn = eSCN(
            num_atoms=0,
            bond_feat_dim=0,
            num_targets=0,
            use_pbc=use_pbc,
            regress_forces=regress_forces,
            regress_stress=regress_stress,
            outer_product_stress=outer_product_stress,
            decomposition_stress=decomposition_stress,
            edge_level=edge_level,
            mixing_coordinates=mixing_coordinates,
            extensive_energy=extensive_energy,
            extensive_stress=extensive_stress,
            otf_graph=otf_graph,
            max_neighbors=max_neighbors,
            cutoff=radius,
            max_num_elements=max_num_elements,
            num_layers=num_layers,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            sphere_channels=sphere_channels,
            hidden_channels=hidden_channels,
            edge_channels=edge_channels,
            use_grid=use_grid,
            num_sphere_samples=num_sphere_samples,
            distance_function=distance_function,
            basis_width_scalar=basis_width_scalar,
            distance_resolution=distance_resolution,
            show_timing_info=show_timing_info,
            enforce_max_neighbors_strictly=enforce_max_neighbors_strictly,
        )
        self.fc_atom = nn.Linear(sphere_channels, MAX_ATOMIC_NUM)

    def forward(self, z, pred_frac_coords, pred_atom_types, num_atoms,
                lengths, angles, conditional=None):
        """
        args:
            z: (N_cryst, num_latent)
            pred_frac_coords: (N_atoms, 3)
            pred_atom_types: (N_atoms, ), need to use atomic number e.g. H = 1
            num_atoms: (N_cryst,)
            lengths: (N_cryst, 3)
            angles: (N_cryst, 3)
        returns:
            atom_frac_coords: (N_atoms, 3)
            atom_types: (N_atoms, MAX_ATOMIC_NUM)
        """
        # (num_atoms, hidden_dim) (num_crysts, 3)
        h, pred_cart_coord_diff, _, _ = self.escn(
            z=z,
            frac_coords=pred_frac_coords,
            atom_types=pred_atom_types,
            num_atoms=num_atoms,
            lengths=lengths,
            angles=angles,
            edge_index=None,
            to_jimages=None,
            num_bonds=None,
            conditional=conditional,
        )
        pred_atom_types = self.fc_atom(h)
        return pred_cart_coord_diff, pred_atom_types
