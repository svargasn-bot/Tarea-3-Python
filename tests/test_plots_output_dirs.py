import os
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')
import pandas as pd

from src.plots import plot_eda_regplots


class PlotOutputDirsTest(unittest.TestCase):
    def test_plot_creates_parent_directory_before_saving(self):
        df = pd.DataFrame(
            {
                'Pre_Semester_GPA': [3.2, 3.5, 3.8, 4.0],
                'Post_Semester_GPA': [3.4, 3.7, 3.9, 4.1],
                'Weekly_GenAI_Hours': [2, 4, 6, 8],
                'Traditional_Study_Hours': [5, 6, 7, 8],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'nested', 'plots', 'eda.png')
            plot_eda_regplots(df, filepath=output_path)
            self.assertTrue(os.path.exists(output_path))


if __name__ == '__main__':
    unittest.main()
