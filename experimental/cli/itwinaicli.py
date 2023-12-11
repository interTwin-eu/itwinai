"""
>>> python itwinaicli.py --config itwinai-conf.yaml --help
>>> python itwinaicli.py --config itwinai-conf.yaml --server.port 333
"""


from itwinai.parser import ConfigParser2
from itwinai.parser import ItwinaiCLI

cli = ItwinaiCLI()
print(cli.pipeline)
print(cli.pipeline.steps)
print(cli.pipeline.steps['server'].port)


parser = ConfigParser2(
    config='itwinai-conf.yaml',
    override_keys={
        'server.init_args.port': 777
    }
)
pipeline = parser.parse_pipeline()
print(pipeline)
print(pipeline.steps)
print(pipeline.steps['server'].port)

server = parser.parse_step('server')
print(server)
print(server.port)
