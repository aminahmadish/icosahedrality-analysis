import sys
import numpy as np
import mdtraj
import itertools
from sklearn.decomposition import PCA
from itertools import combinations

PLANARITY_THRESHOLD = 0.0099 #The lower: the stricter
MIDDLE_BUMP_THRESHOLD = 0.04 #The higher: the stricter
CUTOFF = 0.21
ANGLE_TOLERANCE = 20.0

if len(sys.argv) < 6:
    print('Usage: ' + sys.argv[0] + ' pdb_file selection time_between_frames stride [dcd_file0] [dcd_file1]... output_name')
    sys.exit()

pdb_file = sys.argv[1]
selection = sys.argv[2]
print(selection)
time_between_frames = float(sys.argv[3])
stride = int(sys.argv[4])
dcd_list = sys.argv[5:-1]
output_name = sys.argv[-1]

# If there is no dcd file, just use the PDB we were given
if len(dcd_list) == 0:
    dcd_list = [pdb_file]

print('INPUT:')
print('pdb_file', pdb_file)
print('selection', selection)
print('time_between_frames', time_between_frames)
print('stride', stride)
print('dcd_list', dcd_list)
print('output_name', output_name)
print('PLANARITY_THRESHOLD', PLANARITY_THRESHOLD)
print('CUTOFF', CUTOFF)
print('ANGLE_TOLERANCE', ANGLE_TOLERANCE)

#DEBUGMODE = True
DEBUGMODE = False
PROBIDS = []


def make_pdb_line(serial, name, res_name, pos, beta):
    # Convert to angstroms
    x = 10.0*pos[0]
    y = 10.0*pos[1]
    z = 10.0*pos[2]
    res_id = serial
    return '{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}\n'.format('ATOM', serial, name, '', res_name, 'A', res_id, '', x, y, z, 1.0, beta, name[0])

def wrap(x, box_size):
    return x - box_size*np.floor(x/box_size + 0.5)

def wrap_3d(vec, box_sizes):
    #print(vec, box_sizes)
    return np.array([wrap(vec[0],box_sizes[0]),wrap(vec[1],box_sizes[1]),wrap(vec[2],box_sizes[2])])

# Function to calculate the angle between two vectors
def angle_metric(traj,a,b,c):
    angle = mdtraj.compute_angles(traj, [[a,b,c]], periodic=True)[0]
    return np.degrees(angle)[0]

def check_planarity(this_atom_idx, this_position, posesidx, traj):
    poses = [traj.xyz[0, x] for x in posesidx]

    # Apply periodic boundary conditions
    corrected_poses = []
    for pos in poses:
        displacement = wrap_3d(pos - this_position, traj.unitcell_lengths[0])
        # Apply minimum image convention to displacement vector
        corrected_poses.append(this_position + displacement)

    centroid = np.mean(corrected_poses, axis=0)  # Calculate centroid considering PBC
    centered_positions = [wrap_3d(x - centroid, traj.unitcell_lengths[0]) for x in poses]

    # Apply PCA to find the normal vector to the best-fit plane
    pca = PCA(n_components=3)
    pca.fit(centered_positions)
    norm = pca.components_[2]  # Third component is the plane normal
    
    # Normalize the normal vector
    norm /= np.linalg.norm(norm)
    total_variation = 0

    for p in poses:
        disvec = wrap_3d(p-centroid, traj.unitcell_lengths[0])
        total_variation +=np.abs(np.dot(disvec,norm))
    total_variation/= len(poses)

    # Check the smallest eigenvalue
    if DEBUGMODE:
        if posesidx in PROBIDS:
            print("ID : ", posesidx, "total_variation: ", total_variation)

    if np.isclose(total_variation, 0, atol=PLANARITY_THRESHOLD):
        mid_disvec = wrap_3d(this_position - centroid, traj.unitcell_lengths[0])

        mid_distance = np.abs(np.dot(mid_disvec, norm))
        if DEBUGMODE:
            if posesidx in PROBIDS:
                print("ID : ", posesidx, "mid_distance: ", mid_distance)
    
        #print(f"Middle point distance to plane: {mid_distance}")
        if mid_distance > MIDDLE_BUMP_THRESHOLD: # Thershold middle atom to distance cutoff threshold 
            return True
        else:
            if DEBUGMODE:
                print(f"ELIMINATED ATOM {this_atom_idx} BY BUMP FOR CYCLE {posesidx} AT DISTANCE {mid_distance}")
            return False


def find_pentacycles(edges):
    all_cycles = []

    # Generate all possible combinations of 5 edges from the list of edges
    for subset in combinations(edges, 5):
        # Build a path from these edges to see if they form a cycle of five unique nodes
        graph = {}
        for u, v in subset:
            graph.setdefault(u, []).append(v)
            graph.setdefault(v, []).append(u)

        # Check if this subset forms a five-node cycle
        def dfs(node, start_node, path, visited, depth):
            if depth == 5:
                # Check if it connects back to the start to form a cycle
                if path[0] in graph[node] and len(visited) == 5:
                    all_cycles.append(path[:])  # Add a copy of the cycle
                return

            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, start_node, path, visited, depth + 1)
                    path.pop()
                    visited.remove(neighbor)

        # Attempt to find cycles starting from each node in the subset
        nodes = list(graph.keys())
        for start_node in nodes:
            dfs(start_node, start_node, [start_node], {start_node}, 1)

    # Remove duplicates by treating cycles as sets of nodes (since order doesn't matter in a cycle)
    unique_cycles = set(tuple(sorted(cycle)) for cycle in all_cycles)
    return unique_cycles

def find_icosahedrals(select_idx, traj_frame):
    # Check for icosahedrality
    icosahedral_atoms = []
    involved_atoms = set()

    for i in select_idx:
        this_position = traj_frame.xyz[0,i]
        
        # Find all atoms within CUTOFF Å of atom_i (excluding atom_i), considering periodic boundary conditions
        nearby_atoms = mdtraj.compute_neighbors(traj_frame, CUTOFF, np.array([i]), periodic=True)[0]
        nearby_atoms = [j for j in nearby_atoms if j != i]

        # We can quit now since this atom has fewer than 5 neighbors
        if len(nearby_atoms) < 5:
            #print(f'Atom {i} nothing nearby')
            continue
        
        # Create angle pair list 
        equilateral_count = 0
        angle_pairs = []
        # Loop over all unique pairs of neighbors
        for j in range(len(nearby_atoms)):
            for k in range(j + 1, len(nearby_atoms)):
                neighbor_j = nearby_atoms[j]
                neighbor_k = nearby_atoms[k]

                # Calculate the angle j-i-k
                angle = angle_metric(traj_frame,neighbor_j,i,neighbor_k)
                if DEBUGMODE:
                    if i in PROBIDS:
                        print("ID : ", i, f"abs(angle({neighbor_j},{neighbor_k}) - 60) : ", abs(angle - 60))
    
                # If the angle is within tolerance of 60 degrees
                if abs(angle - 60.0) <= ANGLE_TOLERANCE:
                    equilateral_count += 1
                    angle_pairs.append([neighbor_j,neighbor_k])

        # Check if we don't have at least five triangles, we can't have a pentacycle
        planar_cycles = []
        if equilateral_count >= 5:
            cycles = find_pentacycles(angle_pairs)
            #print(i, cycles)
            planar_cycles = [c for c in cycles if check_planarity(i, this_position, c, traj_frame)]
        
        if DEBUGMODE:
            if i in PROBIDS:
                print("ID : ", i, "involved_atoms : ", involved_atoms, "Path : ", cycle)

        # Is this atom icosahedral
        if len(planar_cycles) != 0:
            icosahedral_atoms.append(i) 
            cycle_set = {item for sublist in planar_cycles for item in sublist}
            cycle_set.add(i)
            involved_atoms.update(cycle_set)

    return icosahedral_atoms, list(involved_atoms)


######################################################################
## Main part of code
with open(f"{output_name}.dat", "w") as output_file:
    traj = mdtraj.load(dcd_list, top=pdb_file, stride=stride)
    print(f"Loaded trajectory {dcd_list} with {traj.n_frames} frames.")

    # Make the selection
    select_idx = traj.topology.select(selection)
    print(f"Selected {len(select_idx)} atoms.")
    
    # Loop over the dcd frames
    for frame_idx, traj_frame in enumerate(traj):
        print(f"Frame {frame_idx} ", end="", flush=True)
        icosahedral_centers, icosahedral_involved = find_icosahedrals(select_idx, traj_frame)
        icosahedral_count = len(icosahedral_centers)
        icosahedral_frac = icosahedral_count/len(select_idx)

        time = frame_idx * time_between_frames * stride
        output_file.write(f"{time:.4f} {icosahedral_count}\n")
        output_file.flush()            
        print(f"count {icosahedral_count} complete {100.0*(frame_idx+1)/traj.n_frames:.2f}%")

# Write a marked PDB for the last frame
with open(f"{output_name}.pdb", "w") as out:
    # The last set of cell basis vectors
    lx = traj.unitcell_lengths[-1, 0]
    ly = traj.unitcell_lengths[-1, 1]
    lz = traj.unitcell_lengths[-1, 2]
    # This directly modifies the variables used in the CRYST1 line.                                            
    lx /= 0.1
    ly /= 0.1
    lz /= 0.1    
    out.write(f'CRYST1{lx:9.3f}{ly:9.3f}{lz:9.3f} 90.00  90.00  90.00 P 1           1\n')
    for i in range(traj.n_atoms):
        # The last frame
        pos = traj.xyz[-1,i]
        if i in icosahedral_centers:
            beta = 2
        elif i in icosahedral_involved:
            beta = 1
        else:
            beta = 0
        out.write(make_pdb_line(i+1, 'B', 'B', pos, beta))
        
