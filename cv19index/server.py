import tempfile
import time
import pandas as pd
import numpy as np

from pkg_resources import resource_filename
from flask import Flask, make_response, request, abort

from .predict import do_run
from .io import read_model


def create_model_app(model_fpath, schema_fpath, **kwargs):
    model = read_model(model_fpath)

    app = Flask(__name__)

    @app.route("/ping", methods=("GET",))
    def ping():
        return make_response("")

    @app.route("/invocations", methods=("POST",))
    def invocations():
        start = time.time()

        with tempfile.NamedTemporaryFile(suffix=".csv") as input_fobj:
            input_fpath = input_fobj.name

            input_fobj.write(request.data)
            input_fobj.seek(0)

            with tempfile.NamedTemporaryFile(suffix=".csv") as output_fobj:
                output_fpath = output_fobj.name

                try:
                    do_run(input_fpath, schema_fpath, model_fpath, output_fpath)
                except Exception as e:
                    abort(400, f"Error parsing payload: {e}")

                output_fobj.seek(0)
                results = output_fobj.read()
                response = make_response(results)
                response.mimetyhpe = "text/csv"
                return response

    return app


def sagemaker_serve():
    app = create_model_app(
        resource_filename("cv19index", "resources/xgboost/model.pickle"),
        resource_filename("cv19index", "resources/xgboost/input.csv.schema.json"),
    )
    app.run("0.0.0.0", 8080, debug=False)
