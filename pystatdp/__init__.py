# MIT License
#
# Copyright (c) 2018 Yuxin Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from jsonpickle import encode
import json
import time
from textwrap import dedent
import argparse
import logging
import multiprocessing as mp
from tqdm import tqdm

from pystatdp.generators import generate_arguments, generate_databases, ALL_DIFFER, ONE_DIFFER
from pystatdp.hypotest import hypothesis_test
from pystatdp.selectors import select_event
from pystatdp.algorithms import generic_method_pydp
from pystatdp.utils import arr_n_check

logger = logging.getLogger(__name__)

matplotlib.use('agg')
matplotlib.rcParams['xtick.labelsize'] = '12'
matplotlib.rcParams['ytick.labelsize'] = '12'


class pystatdp:

    def plot_result(self, data, xlabel, ylabel, title, output_filename):
        """plot the results similar to the figures in our paper
        :param data: The input data sets to plots. e.g., {algorithm_epsilon: [(test_epsilon, pvalue), ...]}
        :param xlabel: The label for x axis.
        :param ylabel: The label for y axis.
        :param title: The title of the figure.
        :param output_filename: The output file name.
        :return: None
        """
        # setup the figure
        plt.ylim(0.0, 1.0)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        # colors and markers for each claimed epsilon
        markers = ['s', 'o', '^', 'x', '*', '+', 'p']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        # add an auxiliary line for p-value=0.05
        plt.axhline(y=0.05, color='black', linestyle='dashed', linewidth=1.2)
        for i, (epsilon, points) in enumerate(data.items()):
            # add an auxiliary vertical line for the claimed privacy
            plt.axvline(x=float(epsilon), color=colors[i % len(
                colors)], linestyle='dashed', linewidth=1.2)
            # plot the
            x = [item[0] for item in points]
            p = [item[1] for item in points]
            plt.plot(x, p, 'o-',
                     label=f'$\\epsilon_0$ = {epsilon}', markersize=8, marker=markers[i % len(markers)], linewidth=3)

        # plot legends
        legend = plt.legend()
        legend.get_frame().set_linewidth(0.0)

        # save the figure and clear the canvas for next draw
        plt.savefig(output_filename, bbox_inches='tight')
        plt.gcf().clear()

    def detect_counterexample(self, algorithm, test_epsilon, default_kwargs=None, databases=None, num_input=(5, 10),
                              event_iterations=100000, detect_iterations=500000, cores=None, sensitivity=ALL_DIFFER,
                              quiet=False, loglevel=logging.INFO):
        """
        :param algorithm: The algorithm to test for.
        :param test_epsilon: The privacy budget to test for, can either be a number or a tuple/list.
        :param default_kwargs: The default arguments the algorithm needs except the first Queries argument.
        :param databases: The databases to run for detection, optional.
        :param num_input: The length of input to generate, not used if database param is specified.
        :param event_iterations: The iterations for event selector to run.
        :param detect_iterations: The iterations for detector to run.
        :param cores: The number of max processes to set for multiprocessing.Pool(), os.cpu_count() is used if None.
        :param sensitivity: The sensitivity setting, all queries can differ by one or just one query can differ by one.
        :param quiet: Do not print progress bar or messages, logs are not affected.
        :param loglevel: The loglevel for logging package.
        :return: [(epsilon, p, d1, d2, kwargs, event)] The epsilon-p pairs along with databases/arguments/selected event.
        """
        # initialize an empty default kwargs if None is given
        default_kwargs = default_kwargs if default_kwargs else {}

        logging.basicConfig(level=loglevel)
        logger.info(
            f'Start detection for counterexample on {algorithm.__name__} with test epsilon {test_epsilon}')
        logger.info(
            f'Options -> default_kwargs: {default_kwargs} | databases: {databases} | cores:{cores}')

        input_list = []
        if databases is not None:
            d1, d2 = databases
            kwargs = generate_arguments(
                algorithm, d1, d2, default_kwargs=default_kwargs)
            input_list = ((d1, d2, kwargs),)
        else:
            num_input = (int(num_input), ) if isinstance(
                num_input, (int, float)) else num_input
            for num in num_input:
                input_list.extend(
                    generate_databases(algorithm, num, default_kwargs=default_kwargs, sensitivity=sensitivity))

        result = []

        # convert int/float or iterable into tuple (so that it has length information)
        test_epsilon = (test_epsilon, ) if isinstance(
            test_epsilon, (int, float)) else test_epsilon

        with mp.Pool(cores) as pool:
            for _, epsilon in tqdm(enumerate(test_epsilon), total=len(test_epsilon), unit='test', desc='Detection',
                                   disable=quiet):
                d1, d2, kwargs, event = select_event(algorithm, input_list, epsilon, event_iterations, quiet=quiet,
                                                     process_pool=pool)
                p = hypothesis_test(algorithm, d1, d2, kwargs, event, epsilon, detect_iterations, report_p2=False,
                                    process_pool=pool)
                if type(d1) != list:
                    d1 = d1.tolist()
                    d2 = d2.tolist()
                result.append((epsilon, float(p), d1, d2, kwargs, event))
                if not quiet:
                    tqdm.write(
                        f'Epsilon: {epsilon} | p-value: {p:5.3f} | Event: {event}')
                logger.debug(f'D1: {d1} | D2: {d2} | kwargs: {kwargs}')

            return result

    def main(self, algo, param, privacy, e_iter=100000, d_iter=500000, test_range=0.1, n_checks=3):
        # list of tasks to test, each tuple contains (function, extra_args, sensitivity)
        tasks = [
            (generic_method_pydp, {'algorithm': algo,
                                   'param_for_algorithm': param}, ALL_DIFFER)
        ]

        # claimed privacy level to check
        claimed_privacy = privacy

        for i, (algorithm, kwargs, sensitivity) in enumerate(tasks):
            start_time = time.time()
            results = {}
            flag_file = time.ctime().replace(' ', '_')
            for privacy_budget in claimed_privacy:
                # # privacy levels to test, here we test the claimed privacy plus .01 above and below
                test_privacy = arr_n_check(privacy_budget, test_range, n_checks)
                # set the second argument of the function (assumed to be `epsilon`) to the claimed privacy level
                kwargs[algorithm.__code__.co_varnames[1]] = privacy_budget
                results[privacy_budget] = self.detect_counterexample(
                    algorithm, test_privacy, kwargs, sensitivity=sensitivity, event_iterations=e_iter, detect_iterations=d_iter)

            # dump the results to file
            json_file = Path.cwd() / f'{algorithm.__name__}_{flag_file}.json'

            with json_file.open('w') as f:
                json.dump(encode(results, unpicklable=False), f)

            # plot and save to file
            plot_file = Path.cwd() / f'{algorithm.__name__}_{flag_file}.pdf'

            self.plot_result(results, r'Test $\epsilon$', 'P Value',
                             algorithm.__name__.replace('_', ' ').title(), plot_file)

            total_time, total_detections = time.time() - start_time, len(claimed_privacy) * \
                len(test_privacy)
            logger.info(f'[{i + 1} / {len(tasks)}]: {algorithm.__name__} | Time elapsed: {total_time:5.3f}s | '
                        f'Average time per detection: {total_time / total_detections:5.3f}s')
        return 0
