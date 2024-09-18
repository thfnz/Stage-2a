import os
import copy
import gc
import matplotlib.pyplot as plt
import numpy as np
import sys

from assistFunct import check_images_dir
from plotResults import plotResults
from assistPlot import assistPlot
from twoSteps import plot_top_n_accuracy, plot_nb_already_labeled
sys.path.append('../')
from logs import logs

def plot_hist_n_top_acc(n_top_accs, reg_stra, n_top, name, folder = '', display = False, save = False):
	bins = np.arange(0, 61, 5)
	plt.figure()
	plt.hist(n_top_accs, bins = bins)
	plt.title('Histogram of the accuracy for the ' + str(n_top) + ' highest target values.\nμ = ' + str(np.mean(n_top_accs)) + ' - σ = ' + str(np.std(n_top_accs)))

	if display:
		plt.show()

	if save:
		check_images_dir('images/' + folder)
		path = './images/'+ folder + 'plot_hist_' + name + '_'
		for stra in reg_stra:
			if type(stra) == list:
				stra = stra[0]
			path += (stra + '_')
		plt.savefig(path + '.png', dpi=300)

	plt.close()

class comparisonAlProcessBaseline2steps:

	def __init__(self, alProcess, alProcess2steps, baseline, X, y, reg_stra, nb_members, n_init, folder = ''):
		self.alProcess = alProcess
		self.alProcess2steps = alProcess2steps
		self.baseline = baseline
		self.X = X
		self.y = y
		self.reg_stra = reg_stra
		self.nb_members = nb_members
		self.n_init = n_init
		self.folder = folder

		if os.path.isdir('./images/' + folder):
			conf = False
			while not conf:
				confirm = input('Save folder (images) already exists, erase previous data ? [y/n]')
				match str(confirm):
					case 'y':
						os.rmdir('./images/' + folder)
						conf = True

					case 'n':
						exit()
						conf = True

		if os.path.isdir('./logs/' + folder):
			conf = False
			while not conf:
				confirm = input('Save folder (logs) already exists, erase previous data ? [y/n]')
				match str(confirm):
					case 'y':
						os.rmdir('./logs/' + folder)
						conf = True

					case 'n':
						exit()
						conf = True

	def comparison_top_n_accuracy(self, nb_processes, pbar = False, display_plot_top_n_accuracy = False, save_plot_top_n_accuracy = False, display_plot_r2 = False, save_plot_r2 = False, lines = 0, columns = 0, display_self_labeled_data_amount = False, save_self_labeled_data_amount = False, display_logs = False, save_logs = False, display = False, save = True):
		self.alProcess_n_top_accs = []
		self.alProcess2steps_n_top_accs = []
		self.alProcess2steps_n_top_train = []
		self.alProcess2steps_n_top_uncertainty = []
		self.baseline_n_top_accs = []

		# Retrieval of the last n_top accuracy after AL
		for idx_process in range(nb_processes):
			al = copy.deepcopy(self.alProcess)
			al2steps = copy.deepcopy(self.alProcess2steps)
			base = copy.deepcopy(self.baseline)

			# Same member_sets initialization
			member_sets, X_test, y_test = al2steps.member_setsInit(self.X, self.y, self.reg_stra, self.nb_members, self.n_init, display = display)
			al.member_sets, al2steps.member_sets, base.member_sets = copy.deepcopy(member_sets), copy.deepcopy(member_sets), copy.deepcopy(member_sets)
			al.X_test, al.y_test = copy.deepcopy(X_test), copy.deepcopy(y_test)
			al.X, al.y, al.reg_stra = self.X, self.y, self.reg_stra
			base.X_test, base.y_test = copy.deepcopy(X_test), copy.deepcopy(y_test)
			base.X, base.y, base.reg_stra = self.X, self.y, self.reg_stra

			print('Iteration #' + str(idx_process + 1))
			al.learn(display = display, pbar = pbar)
			al2steps.learn(display = display, pbar = pbar)
			base.learn(display = display, pbar = pbar)

			self.alProcess_n_top_accs.append(al.class_set[0][-1])
			self.alProcess2steps_n_top_accs.append(al2steps.class_set[0][-1])
			# self.alProcess2steps_n_top_train.append(al2steps.class_set[2][-1])
			self.alProcess2steps_n_top_uncertainty.append(al2steps.class_set[3][-1])
			self.baseline_n_top_accs.append(base.class_set[0][-1])

			# plot_top_n_accuracy
			if display_plot_top_n_accuracy or save_plot_top_n_accuracy or display_plot_r2 or save_plot_r2:
				plot_al = plotResults(al)
				plot_base = plotResults(base)

				if display_plot_top_n_accuracy or save_plot_top_n_accuracy:
					plot_al.top_n_accuracy(name = type(self.alProcess).__name__ + '_alProcess_it' + str(idx_process + 1), folder = self.folder + 'top_n_accuracy/al/', display = display_plot_top_n_accuracy, save = save_plot_top_n_accuracy)
					plot_top_n_accuracy(al2steps, name = type(self.alProcess2steps).__name__ + '_alProcess2steps_it' + str(idx_process + 1), folder = self.folder + 'top_n_accuracy/al2steps/', display = display_plot_top_n_accuracy, save = save_plot_top_n_accuracy)
					plot_nb_already_labeled(al2steps, name = '_' + type(self.alProcess2steps).__name__ + '_alProcess2steps_it' + str(idx_process + 1), folder = self.folder + 'nb_already_labeled/', display = display_plot_top_n_accuracy, save = save_plot_top_n_accuracy)
					if type(self.baseline).__name__ != 'fastRandomQuery':
						plot_base.top_n_accuracy(name = type(self.baseline).__name__ + '_base_it' + str(idx_process + 1), folder = self.folder + 'top_n_accuracy/base/', display = display_plot_top_n_accuracy, save = save_plot_top_n_accuracy)

				if display_plot_r2 or save_plot_r2:
					### TODO : allouer dynamiquement lignes et colonnes
					plot_al.r2(lines, columns, name = 'alProcess_it' + str(idx_process + 1), folder = self.folder + 'r2/al/', display = display_plot_r2, save = save_plot_r2)
					plot_al2steps.r2(lines, columns, name = 'alProcess2steps_it' + str(idx_process + 1), folder = self.folder + 'r2/al2steps/', display = display_plot_r2, save = save_plot_r2)
					if type(self.baseline).__name__ != 'fastRandomQuery':
						plot_base.r2(lines, columns, name = 'base_it' + str(idx_process + 1), folder = self.folder + 'r2/base/', display = display_plot_r2, save = save_plot_r2)

				del plot_al
				del plot_base

			if display_self_labeled_data_amount or save_self_labeled_data_amount:
				assistPlot_al = assistPlot(al)
				assistPlot_al2steps = assistPlot(al2steps)

				assistPlot_al.self_labeled_data_amount(idx = 3, name = 'it' + str(idx_process + 1), folder = self.folder + 'sld_amount/al/', display = display_self_labeled_data_amount, save = save_self_labeled_data_amount)
				assistPlot_al2steps.self_labeled_data_amount(idx = 4, name = 'it' + str(idx_process + 1), folder = self.folder + 'sld_amount/al2steps/', display = display_self_labeled_data_amount, save = save_self_labeled_data_amount)
				del assistPlot_al

			if display_logs or save_logs:
				### TODO : display_logs
				logs_al = logs(al)
				logs_al2steps = logs(al2steps)
				if type(self.baseline).__name__ != 'fastRandomQuery':
					logs_base = logs(base)

				if save_logs:
					logs_al.gen_save(n_init = self.n_init, twoSteps = False, last_instance_added = True, r2 = True, n_top = True, name = 'it' + str(idx_process + 1), folder = self.folder + 'al/')
					logs_al2steps.gen_save(n_init = self.n_init, twoSteps = True, last_instance_added = True, r2 = True, n_top = True, name = 'it' + str(idx_process + 1), folder = self.folder + 'al2steps/')
					if type(self.baseline).__name__ != 'fastRandomQuery':
						logs_base.gen_save(n_init = self.n_init, twoSteps = False, last_instance_added = True, r2 = True, n_top = True, name = 'it' + str(idx_process + 1), folder = self.folder + 'base/')

			del al
			del base
			gc.collect()

		# Plot
		plot_hist_n_top_acc(np.array(self.alProcess_n_top_accs), self.reg_stra, self.alProcess.n_top, 'alProcess_n_top_accs', folder = self.folder, display = display, save = save)
		plot_hist_n_top_acc(np.array(self.alProcess2steps_n_top_accs), self.reg_stra, self.alProcess2steps.n_top, 'alProcess2steps_n_top_accs', folder = self.folder, display = display, save = save)
		# plot_hist_n_top_acc(np.array(self.alProcess2steps_n_top_train), self.reg_stra, self.alProcess2steps.n_top, 'alProcess2steps_n_top_train', folder = self.folder, display = display, save = save)
		plot_hist_n_top_acc(np.array(self.alProcess2steps_n_top_uncertainty), self.reg_stra, self.alProcess2steps.n_top, 'alProcess2steps_n_top_uncertainty', folder = self.folder, display = display, save = save)
		plot_hist_n_top_acc(np.array(self.baseline_n_top_accs), self.reg_stra, self.baseline.n_top, 'baseline_n_top_accs', folder = self.folder, display = display, save = save)


