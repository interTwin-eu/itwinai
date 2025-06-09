#!/usr/bin/env python3
# before it was !/opt/conda/bin/python
# -*- coding: utf-8 -*-
#
# D. Ciangottini
#
import asyncio
import os
import re
import sys
from subprocess import Popen

import asyncssh
from jupyterhub.singleuser import main

ssh_host = os.environ.get("JHUB_HOST")
ssh_url_port = os.environ.get("SSH_PORT")
username = os.environ.get("JUPYTERHUB_USER")
token = os.environ.get("JUPYTERHUB_API_TOKEN")

fwd_port = os.environ.get("FWD_PORT")


async def run_client():
    async with asyncssh.connect(
        host=ssh_host,
        port=int(ssh_url_port),
        username=username,
        password=token,
        known_hosts=None,
    ) as conn:
        conn.set_keepalive(interval=14.0, count_max=10)
        listener = await conn.forward_remote_port(
            "0.0.0.0",
            int(fwd_port),
            "0.0.0.0",
            int(fwd_port),
        )
        await listener.wait_closed()


if __name__ == "__main__":
    print("Connecting ssh...")
    loop = asyncio.get_event_loop()
    loop.create_task(run_client())

    print("Configuring Rucio extension...")
    p = Popen(["/usr/local/bin/setup.sh"])
    while p.poll() is None:
        pass

    print("Starting JLAB")
    sys.argv[0] = re.sub(r"(-script\.pyw|\.exe)?$", "", sys.argv[0])
    sys.exit(main())
