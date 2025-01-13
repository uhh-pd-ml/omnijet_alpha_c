import awkward as ak
import numpy as np
import vector

vector.register_awkward()


def merge_duplicates_numpy_old(rounded_showers):
    """Merges duplicate voxels using NumPy for intermediate processing.

    Args:
        rounded_showers: An Awkward Array of showers with voxel
                         coordinates.

    Returns:
        An Awkward Array with duplicate voxels merged (energies summed).
    """

    merged_showers = []
    for shower in rounded_showers:
        # Convert to NumPy arrays
        x = np.round(shower.x.to_numpy()).astype(int)
        y = np.round(shower.y.to_numpy()).astype(int)
        z = np.round(shower.z.to_numpy()).astype(int)
        energy = shower.energy.to_numpy()

        # Create a structured array for easier duplicate handling
        voxel_ids = np.stack((x, y, z), axis=-1)

        # Find unique voxels and their indices
        unique_voxels, inverse_indices = np.unique(voxel_ids, return_inverse=True, axis=0)

        # Sum energies of duplicate voxels
        max_energy = np.zeros(len(unique_voxels))
        np.maximum.at(max_energy, inverse_indices, energy)

        # Construct updated shower with merged energies
        merged_shower = ak.zip(
            {
                "x": unique_voxels[:, 0],
                "y": unique_voxels[:, 1],
                "z": unique_voxels[:, 2],
                "energy": max_energy,
            },
            with_name="data",
        )

        merged_showers.append(merged_shower)

    return ak.Array(merged_showers)


def merge_duplicates_numpy(rounded_showers):
    """Merges duplicate voxels using NumPy for intermediate processing and takes the maximum
    energy.

    Args:
        rounded_showers: An Awkward Array of showers with voxel
                         coordinates.

    Returns:
        An Awkward Array with duplicate voxels merged (maximum energy taken).
    """

    merged_showers = []
    for shower in rounded_showers:
        # Convert to NumPy arrays
        x = np.round(shower.x.to_numpy()).astype(int)
        y = np.round(shower.y.to_numpy()).astype(int)
        z = np.round(shower.z.to_numpy()).astype(int)
        energy = shower.energy.to_numpy()

        # Create a structured array for easier duplicate handling
        voxel_ids = np.stack((x, y, z), axis=-1)

        # Find unique voxels and their indices
        unique_voxels, inverse_indices = np.unique(voxel_ids, return_inverse=True, axis=0)

        # Max energies of duplicate voxels
        # max_energy = np.zeros(len(unique_voxels))
        # np.maximum.at(max_energy, inverse_indices, energy)

        # Shift duplicate voxels by z-layer if spot is free
        for idx in range(len(voxel_ids)):
            if np.sum(inverse_indices == idx) > 1:  # Check for duplicates
                # Find the unique voxel for this duplicate
                unique_voxel = unique_voxels[inverse_indices[idx]]

                # Find a free spot in the z-direction and shift the duplicate
                # Find the unique voxel for this duplicate
                unique_voxel = unique_voxels[inverse_indices[idx]]

                # Get the energies of the duplicate voxels
                duplicate_indices = np.where(inverse_indices == inverse_indices[idx])[0]
                duplicate_energies = energy[duplicate_indices]

                # Find the index of the voxel with the highest energy
                max_energy_index = duplicate_indices[np.argmax(duplicate_energies)]

                # Shift all other duplicate voxels
                for duplicate_idx in duplicate_indices:
                    if duplicate_idx != max_energy_index:
                        shift = 1
                        while True:
                            # Check for free spot in +z direction
                            if not np.any(
                                (voxel_ids == (unique_voxel + [0, 0, shift])).all(axis=1)
                            ):
                                voxel_ids[duplicate_idx] = unique_voxel + [0, 0, shift]
                                break

                            # Check for free spot in -z direction
                            if not np.any(
                                (voxel_ids == (unique_voxel + [0, 0, -shift])).all(axis=1)
                            ):
                                voxel_ids[duplicate_idx] = unique_voxel + [0, 0, -shift]
                                break

                            shift += 1  # Increment the shift for next iteration

        # Construct updated shower with maximum energies
        merged_shower = ak.zip(
            {
                "x": voxel_ids[:, 0],
                "y": voxel_ids[:, 1],
                "z": voxel_ids[:, 2],
                "energy": energy,
            },
            with_name="data",
        )

        merged_showers.append(merged_shower)

    return ak.Array(merged_showers)
