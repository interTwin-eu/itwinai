from itwinai.parser import ConfigParser

if __name__ == "__main__":

    parser = ConfigParser(
        config="config.yaml"
    )
    my_pipeline = parser.parse_pipeline(
        pipeline_nested_key="training_pipeline",
        verbose=False
    )
    print("Pipeline parsed successfully!")
    # Skip the first step of the pipeline (data generation)
    my_pipeline.execute()
