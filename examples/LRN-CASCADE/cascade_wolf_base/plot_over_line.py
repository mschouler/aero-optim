import argparse
import os

import paraview.simple as pvs


def main():
    """
    This program extracts data along the inlet/outlet or measurement plans
    of the cascade experiment. It produces two .csv files respectively corresponding
    to the data at the inlet (or MP1) and at the outlet (or MP2).
    - Vector2:0 = rho*u
    - Vector2:1 = rho*v
    - Scalar1_input_1 = P
    - Scalar2 = P0

    Note: pres.vtu and final.vtu are expected to be found in the results directory
          these can be produced from wolf results by running
          $ gmf2vtk -in <file>.mesh(b) -sol <file>.solb -out <file>.vtu

    Warning: this program must be executed with pvpython
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-in", "--input", type=str, help="results directory", default=os.getcwd())
    parser.add_argument(
        "-mp", "--mplan", action='store_true', help="extract results along MP1 and MP2"
    )
    args = parser.parse_args()
    n_sampling = 1000

    # disable automatic camera reset on 'Show'
    pvs._DisableFirstRenderCameraReset()

    # create a new 'XML Unstructured Grid Reader'
    presvtu = pvs.XMLUnstructuredGridReader(
        registrationName='pres.vtu',
        FileName=[os.path.join(args.input, 'pres.vtu')]  # noqa
    )
    finalvtu = pvs.XMLUnstructuredGridReader(
        registrationName='final.vtu',
        FileName=[os.path.join(args.input, 'final.vtu')]  # noqa
    )

    # Properties modified on presvtu
    presvtu.TimeArray = 'None'
    finalvtu.TimeArray = 'None'

    # pvs.UpdatePipeline(time=0.0, proxy=presvtu)
    # create a new 'Group Datasets'
    groupDatasets1 = pvs.AppendAttributes(
        registrationName='GroupDatasets1', Input=[finalvtu, presvtu]
    )

    pvs.UpdatePipeline(time=0.0, proxy=groupDatasets1)

    # get active source.
    groupDatasets1 = pvs.GetActiveSource()

    # set active source
    pvs.SetActiveSource(groupDatasets1)

    # create a new 'Plot Over Line'
    plotOverLine1 = pvs.PlotOverLine(registrationName='PlotOverLine1', Input=groupDatasets1)

    # Properties modified on plotOverLine1
    plotOverLine1.SamplingPattern = 'Sample Uniformly'  # 'Sample At Cell Boundaries'
    plotOverLine1.Resolution = n_sampling

    if args.mplan:
        plotOverLine1.Point1 = [-0.02, -0.02778, 0.0]
        plotOverLine1.Point2 = [-0.02, 0.01261, 0.0]
    else:
        plotOverLine1.Point1 = [-0.06, -0.05, 0.0]
        plotOverLine1.Point2 = [-0.06, -9.61e-3, 0.0]

    pvs.UpdatePipeline(time=0.0, proxy=plotOverLine1)

    # save data
    pvs.SaveData(
        f"plot{'MP' if args.mplan else ''}1.csv",
        proxy=plotOverLine1,
        PointDataArrays=[' Scalar1', ' Scalar2', ' Vector2', 'Reference', 'arc_length', 'vtkValidPointMask'] # noqa
    )

    # go back to main source
    pvs.UpdatePipeline(time=0.0, proxy=groupDatasets1)

    # create a new 'Plot Over Line'
    plotOverLine2 = pvs.PlotOverLine(registrationName='PlotOverLine2', Input=groupDatasets1)

    # Properties modified on plotOverLine1
    plotOverLine2.SamplingPattern = 'Sample Uniformly'  # 'Sample At Cell Boundaries'
    plotOverLine2.Resolution = n_sampling

    if args.mplan:
        plotOverLine2.Point1 = [0.087, 0.0, 0.0]
        plotOverLine2.Point2 = [0.087, 4.039e-2, 0.0]
    else:
        plotOverLine2.Point1 = [0.13, 0.0, 0.0]
        plotOverLine2.Point2 = [0.13, 4.039e-2, 0.0]

    pvs.UpdatePipeline(time=0.0, proxy=plotOverLine2)

    # save data
    pvs.SaveData(
        f"plot{'MP' if args.mplan else ''}2.csv",
        proxy=plotOverLine2,
        PointDataArrays=[' Scalar1', ' Scalar2', ' Vector2', 'Reference', 'arc_length', 'vtkValidPointMask'] # noqa
    )


if __name__ == "__main__":
    main()
