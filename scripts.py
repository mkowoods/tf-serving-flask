import subprocess
import time

command = """
cd /home/mkowoods/tf-serving-flask \
&& sudo git fetch origin \
&& sudo git reset --hard origin/master \
&& echo restarting tf_serving \
&& sudo systemctl restart tf_serving \
&& echo restarting tf_serving_app \
&& sudo systemctl restart tf_serving_app \
&& echo restarting nginx \
&& sudo nginx -s reload \
"""

ssh = " gcloud compute ssh --zone {zone} {instance} --command '{command}' "

instances = [
    ('us-west1-a', 'tf-serving-1'),
    ('us-west1-b', 'tf-serving-2'),
    ('us-west1-c', 'tf-serving-3'),
    ('us-west1-a', 'tf-serving-4'),
]
for zone, instance in instances[:1]:
    print zone, instance
    print subprocess.call(ssh.format(zone=zone, instance=instance, command=command), shell=True)
    time.sleep(2)