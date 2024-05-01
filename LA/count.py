import argparse
import subprocess
from collections import defaultdict

NUM_GPUS_PER_NODE = 8
NUM_CPUS_PER_NODE = 256
NUM_MEM_PER_NODE = 1031  # GB


def get_num_nodes(args):
    result = subprocess.run(
        f"sinfo -hp {args.partition}".split(), stdout=subprocess.PIPE
    )
    result = result.stdout.decode("utf-8")
    lines = result.strip().split("\n")

    num_nodes = 0
    for line in lines:
        parts = line.split()
        avail = parts[1]
        nodes = int(parts[3])
        state = parts[4]

        num_nodes += int(avail == "up" and state in ["alloc", "idle", "mix"]) * nodes

    return num_nodes


def get_gpus_per_user(args):
    gpus_per_user = defaultdict(lambda: {"running": 0, "pending": 0})

    # Get pending GPUs
    result = subprocess.run(
        f"squeue -h -t PD -p {args.partition} -O  JobID,UserName,NumNodes,tres-per-node,tres-per-job".split(),
        stdout=subprocess.PIPE,
    )
    result = result.stdout.decode("utf-8")
    lines = result.strip().split("\n")
    for line in lines:
        if line:
            parts = line.split()
            user = parts[1]
            num_nodes = int(parts[2])
            tres_per_node = int(parts[3].split(":")[-1]) if parts[3] != "N/A" else 0
            tres_per_job = int(parts[4].split(":")[-1]) if parts[4] != "N/A" else 0

            pending_gpus = num_nodes * tres_per_node + tres_per_job
            gpus_per_user[user]["pending"] += pending_gpus

    # Get running GPUs
    result = subprocess.run(
        f"squeue -h -t R -p {args.partition} -O JobID,UserName,NumNodes,tres-per-node,tres-per-job".split(),
        stdout=subprocess.PIPE,
    )
    result = result.stdout.decode("utf-8")
    lines = result.strip().split("\n")
    for line in lines:
        if line:
            parts = line.split()
            user = parts[1]
            num_nodes = int(parts[2])
            tres_per_node = int(parts[3].split(":")[-1]) if parts[3] != "N/A" else 0
            tres_per_job = int(parts[4].split(":")[-1]) if parts[4] != "N/A" else 0

            running_gpus = num_nodes * tres_per_node + tres_per_job
            gpus_per_user[user]["running"] += running_gpus

    gpus_per_user = {
        k: v
        for k, v in sorted(
            gpus_per_user.items(), key=lambda item: item[1]["running"], reverse=True
        )
    }
    return gpus_per_user


def get_remain_per_node(args):
    result = subprocess.run(
        f"scontrol -o show nodes | grep {args.partition} | awk '{{ print $1, $38}}'",
        stdout=subprocess.PIPE,
        shell=True,
    )
    result = result.stdout.decode("utf-8")
    lines = result.strip().split("\n")

    unavailable_nodes = subprocess.run(
        f"sinfo -hp {args.partition} | grep -E \"(down|drain)\" | awk '{{print $6}}'",
        stdout=subprocess.PIPE,
        shell=True,
    )
    unavailable_nodes = unavailable_nodes.stdout.decode("utf-8")
    unavailable_nodes = unavailable_nodes.strip()

    remain_per_node = {}
    for line in lines:
        node, alloc = line.split()
        _, node_name = node.split("=")
        node_num = node_name[node_name.rfind("-") + 1 :]
        if node_num not in unavailable_nodes:
            remain_resources = [NUM_CPUS_PER_NODE, NUM_MEM_PER_NODE, NUM_GPUS_PER_NODE]
            for resource in alloc.split(","):
                current_alloc = resource[resource.rfind("=") + 1 :]
                if "cpu" in resource:
                    remain_resources[0] -= int(current_alloc)
                elif "mem" in resource:
                    unit = current_alloc[-1]
                    current_alloc = int(float(current_alloc[:-1]))
                    if unit == "T":
                        current_alloc *= 1e3
                    elif unit == "M":
                        current_alloc //= 1e3
                    elif unit == "K":
                        current_alloc //= 1e6
                    remain_resources[1] -= int(current_alloc)
                elif "gpu" in resource:
                    remain_resources[2] -= int(current_alloc)

            if sum(remain_resources):
                remain_per_node[node_name] = remain_resources

    return remain_per_node


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--partition", type=str, default="research")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-r", "--resources", action="store_true")

    args = parser.parse_args()

    if not args.resources:
        num_nodes = get_num_nodes(args)
        gpus_per_user = get_gpus_per_user(args)
        total_used_gpus = sum([gpus["running"] for gpus in gpus_per_user.values()])

        if args.verbose:
            print(f"{'USER':<16} {'RUNNING_GPUS':<16} PENDING_GPUS")
            for username, gpus in gpus_per_user.items():
                print(f"{username:<16} {gpus['running']:<16} {gpus['pending']}")

        print(f"\nTotal used GPUs: {total_used_gpus}/{num_nodes * NUM_GPUS_PER_NODE}")
        print(
            f"Total utilization: {(total_used_gpus * 100) / (num_nodes * NUM_GPUS_PER_NODE):.2f}%"
        )
    else:
        remain_per_node = get_remain_per_node(args)
        remain_per_node = {
            k: v
            for k, v in sorted(
                remain_per_node.items(), key=lambda item: item[1][-1], reverse=True
            )
        }
        print(f"{'NODE':<24} {'CPUS':<16} {'MEM':<16} GPUS")
        for node, resources in remain_per_node.items():
            print(
                f"{node:<24} {resources[0]:<16} {f'{resources[1]}G':<16} {resources[2]}"
            )