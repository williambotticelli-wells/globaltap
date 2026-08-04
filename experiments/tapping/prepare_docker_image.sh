#!/bin/bash
# Patch repp analysis.py to fix empty-array ValueError in plot_markers_error
# (np.min(t[t < 1000]) fails when t[t < 1000] is empty)
set -e
python3 << 'PATCH_SCRIPT'
import os
import repp
path = os.path.join(os.path.dirname(repp.__file__), 'analysis.py')
with open(path, 'r') as f:
    content = f.read()
if 't_filtered = t[t < 1000]' in content:
    print('Already patched')
    exit(0)
old_plain = """    t = s_marker_ideal[~np.isnan(s_marker_ideal)]
    if len(aligned_onsets['markers_onsets_detected']) > 0 and len(t) > 0:
        plt.plot(aligned_onsets['markers_onsets_detected'] / 1000.0,
                 0.95 * np.min(t[t < 1000]) * np.ones(np.size(aligned_onsets['markers_onsets_detected'])), 'sm')
        plt.ylim([0.9 * min(t[t < 1000]), 1.05 * max(t[t < 1000])])"""
new_plain = """    t = s_marker_ideal[~np.isnan(s_marker_ideal)]
    t_filtered = t[t < 1000]
    if len(aligned_onsets['markers_onsets_detected']) > 0 and len(t_filtered) > 0:
        plt.plot(aligned_onsets['markers_onsets_detected'] / 1000.0,
                 0.95 * np.min(t_filtered) * np.ones(np.size(aligned_onsets['markers_onsets_detected'])), 'sm')
        plt.ylim([0.9 * np.min(t_filtered), 1.05 * np.max(t_filtered)])"""
if old_plain in content:
    content = content.replace(old_plain, new_plain)
    with open(path, 'w') as f:
        f.write(content)
    print(f"Patched {path}")
else:
    print(f"Patch not applied - source may have changed. Check {path}")

# Patch psynet experiment.py: pass assignment_id when participant is None (fixes Internal Server Error on terminate_participant)
import psynet
exp_path = os.path.join(os.path.dirname(psynet.__file__), 'experiment.py')
with open(exp_path, 'r') as f:
    content = f.read()
old_route = """    # Lucid recruitment specific route
    @experiment_route("/terminate_participant", methods=["GET"])
    @classmethod
    @with_transaction
    def terminate_participant(cls):
        recruiter = get_experiment().recruiter
        participant = recruiter.get_participant(request)
        external_submit_url = recruiter.terminate_participant(
            participant=participant, reason=request.values.get("reason")
        )

        return render_template_with_translations(
            "exit_recruiter_lucid.html",
            external_submit_url=external_submit_url,
        )"""
new_route = """    # Lucid recruitment specific route
    @experiment_route("/terminate_participant", methods=["GET"])
    @classmethod
    @with_transaction
    def terminate_participant(cls):
        recruiter = get_experiment().recruiter
        participant = recruiter.get_participant(request)
        # Pass assignment_id when participant is None (e.g. wrong-browser before DB record exists)
        assignment_id = (
            request.values.get("assignmentId") or request.values.get("RID")
            if participant is None
            else None
        )
        external_submit_url = recruiter.terminate_participant(
            participant=participant,
            assignment_id=assignment_id,
            reason=request.values.get("reason"),
        )

        return render_template_with_translations(
            "exit_recruiter_lucid.html",
            external_submit_url=external_submit_url,
        )"""
if old_route in content and new_route not in content:
    content = content.replace(old_route, new_route)
    with open(exp_path, 'w') as f:
        f.write(content)
    print(f"Patched {exp_path} (terminate_participant)")
elif 'assignment_id=assignment_id' in content and 'terminate_participant' in content:
    print('PsyNet terminate_participant already patched')
else:
    print(f"PsyNet terminate_participant patch not applied - check {exp_path}")

# Patch dallinger EC2 list instances: pagination (see all instances), .get() for missing fields, tqdm, --all flag
import dallinger
dallinger_dir = os.path.dirname(dallinger.__file__)
ec2_lib_path = os.path.join(dallinger_dir, 'command_line', 'lib', 'ec2.py')
ec2_cli_path = os.path.join(dallinger_dir, 'command_line', 'ec2.py')

if os.path.exists(ec2_lib_path):
    with open(ec2_lib_path, 'r') as f:
        ec2_content = f.read()
    if 'while "NextToken" in response' in ec2_content:
        print('Dallinger EC2 lib already patched')
    else:
        ec2_content = ec2_content.replace(
            'reservations = get_ec2_client(region_name).describe_instances()["Reservations"]',
            '''ec2 = get_ec2_client(region_name)
    reservations = []
    response = ec2.describe_instances()
    reservations.extend(response["Reservations"])
    while "NextToken" in response:
        response = ec2.describe_instances(NextToken=response["NextToken"])
        reservations.extend(response["Reservations"])'''
        )
        ec2_content = ec2_content.replace(
            '"public_dns_name": instance["PublicDnsName"]',
            '"public_dns_name": instance.get("PublicDnsName", "")'
        )
        ec2_content = ec2_content.replace(
            '"pem": instance["KeyName"]',
            '"pem": instance.get("KeyName", "")'
        )
        ec2_content = ec2_content.replace(
            'pb = tqdm(all_regions, total=len(all_regions))',
            '''pb = tqdm(
            all_regions,
            total=len(all_regions),
            mininterval=1.0,
            file=sys.stderr,
            disable=not sys.stderr.isatty(),
        )'''
        )
        if 'import sys' not in ec2_content and 'from sys' not in ec2_content:
            ec2_content = ec2_content.replace('\nimport ', '\nimport sys\nimport ', 1)
            if 'import sys' not in ec2_content:
                ec2_content = ec2_content.replace('import ', 'import sys\nimport ', 1)
        ec2_content = ec2_content.replace(
            '    if pem is not None:\n        instance_df = instance_df.query("pem.str.endswith(@pem)")\n    print(instance_df.to_markdown())',
            '    if pem is not None:\n        instance_df = instance_df.query("pem.str.endswith(@pem)")\n    instance_df = instance_df.reset_index(drop=True)\n    print(instance_df.to_markdown())'
        )
        with open(ec2_lib_path, 'w') as f:
            f.write(ec2_content)
        print(f"Patched {ec2_lib_path} (EC2 list-instances pagination)")
else:
    print(f"Dallinger EC2 lib not found at {ec2_lib_path}")

if os.path.exists(ec2_cli_path):
    with open(ec2_cli_path, 'r') as f:
        ec2_cli_content = f.read()
    if 'all_instances' in ec2_cli_content:
        print('Dallinger EC2 --all flag already patched')
    else:
        ec2_cli_content = ec2_cli_content.replace(
            '@click.option("--terminated", is_flag=True, help="List terminated instances")\n@click.pass_context\ndef list__instances(ctx, region, running, stopped, terminated):',
            '@click.option("--terminated", is_flag=True, help="List terminated instances")\n@click.option("--all", "all_instances", is_flag=True, help="List all instances (ignore PEM/key filter)")\n@click.pass_context\ndef list__instances(ctx, region, running, stopped, terminated, all_instances):'
        )
        ec2_cli_content = ec2_cli_content.replace(
            'pem = cfg.get("pem")',
            'pem = None if all_instances else cfg.get("pem")'
        )
        with open(ec2_cli_path, 'w') as f:
            f.write(ec2_cli_content)
        print(f"Patched {ec2_cli_path} (EC2 list --all flag)")
else:
    print(f"Dallinger EC2 CLI not found at {ec2_cli_path}")
PATCH_SCRIPT
