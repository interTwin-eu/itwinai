"""
The most simple workflow that you can write is a sequential pipeline of steps,
where the outputs of a component are fed as input to the following component,
employing a scikit-learn-like Pipeline.

In itwinai, a step is also called "component" and is implemented by extending
the ``itwinai.components.BaseComponent`` class. Each component implements
the `execute(...)` method, which provides a unified interface to interact with
each component.

The aim of itwinai components is to provide reusable machine learning best
practices, and some common operations are already encoded in some abstract
components. Some examples are:
- ``DataGetter``: has no input and returns a dataset, collected from somewhere
(e.g., downloaded).
- ``DataSplitter``: splits an input dataset into train, validation and test.
- ``DataPreproc``: perform preprocessing on train, validation, and test
datasets.
- ``Trainer``: trains an ML model and returns the trained model.
- ``Saver``: saved an ML artifact (e.g., dataset, model) to disk.

In this tutorial you will see how to create new components and how they
are assembled into sequential pipelines. Newly created components are
in a separate file called 'basic_components.py'.
"""
from itwinai.pipeline import Pipeline

# Import the custom components from file
from basic_components import MyDataGetter, MyDatasetSplitter, MyTrainer

if __name__ == "__main__":
    # Assemble them in a scikit-learn like pipeline
    pipeline = Pipeline([
        MyDataGetter(data_size=100),
        MyDatasetSplitter(
            train_proportion=.5,
            validation_proportion=.25,
            test_proportion=0.25
        ),
        MyTrainer()
    ])

    # Inspect steps
    print(pipeline[0])
    print(pipeline[2].name)
    print(pipeline[1].train_proportion)

    # Run pipeline
    _, _, _, trained_model = pipeline.execute()
    print("Trained model: ", trained_model)

    # You can also create a Pipeline from a dict of components, which
    # simplifies their retrieval by name
    pipeline = Pipeline({
        "datagetter": MyDataGetter(data_size=100),
        "splitter": MyDatasetSplitter(
            train_proportion=.5,
            validation_proportion=.25,
            test_proportion=0.25
        ),
        "trainer": MyTrainer()
    })

    # Inspect steps
    print(pipeline["datagetter"])
    print(pipeline["trainer"].name)
    print(pipeline["splitter"].train_proportion)

    # Run pipeline
    _, _, _, trained_model = pipeline.execute()
    print("Trained model: ", trained_model)
