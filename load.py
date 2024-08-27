import os

import plotData


def load_data(data_dir, subject, session, task_type, sensor_keyword, level, run_label):
    data = plotData.loadTimeSeries(data_dir, subject, session, task_type, sensor_keyword, level + "_" + run_label)
    print("\nData Columns: \n\t" + str(list(data.columns)))
    data_len = len(data['time_dn'])
    print("\nNumber of Time Samples: " + str(data_len))
    return data


if __name__ == '__main__':
    cwd = os.getcwd()
    main_dir = os.path.split(cwd)[0]
    data_dir = os.path.split(main_dir)[0]
    data_pkg_trg_dir = os.path.join(data_dir, 'dataPackage')
    sub022_run001_dfEDA = load_data(data_pkg_trg_dir, "sub-cp022", "ses-20210621", "task-ils", "lslshimmereda", "level-01B", "run-001")

