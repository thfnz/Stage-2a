import copy
import gc
import matplotlib.pyplot as plt

from assistFunct import check_images_dir
from plotResults import plotResults

def plot_hist_n_top_acc(n_top_accs, reg_stra, name, display = False, save = False):
	plt.figure()
	plt.hist(n_top_accs)

	if display:
		plt.show()

	if save:
		check_images_dir('')
		path = './images/plot_hist_' + name + '_'
		for stra in reg_stra:
			path += (stra + '_')
		plt.savefig(path + '.png', dpi=300)

	plt.close()

class comparisonAlProcessBaseline:

	def __init__(self, alProcess, baseline, X, y, reg_stra, nb_members, n_init):
		self.alProcess = alProcess
		self.baseline = baseline
		self.X = X
		self.y = y
		self.reg_stra = reg_stra
		self.nb_members = nb_members
		self.n_init = n_init

	def comparison_top_n_accuracy(self, nb_processes, display_plot_top_n_accuracy = False, save_plot_top_n_accuracy = False, display_plot_r2 = False, save_plot_r2 = False, display = False, save = True, pbar = False):
		self.alProcess_n_top_accs = []
		self.baseline_n_top_accs = []

		# Same member_sets initialization
		member_sets, X_test, y_test = self.alProcess.member_setsInit(self.X, self.y, self.reg_stra, self.nb_members, self.n_init, display = display)
		self.alProcess.member_sets, self.baseline.member_sets = member_sets, member_sets
		self.baseline.X, self.baseline.y, self.baseline.reg_stra = self.X, self.y, self.reg_stra
		self.baseline.X_test, self.baseline.y_test = X_test, y_test

		# Retrieval of the last n_top accuracy after AL
		for idx_process in range(nb_processes):
			al = copy.deepcopy(self.alProcess)
			base = copy.deepcopy(self.baseline)

			al.learn(display = display, pbar = pbar)
			base.learn(display = display, pbar = pbar)

			self.alProcess_n_top_accs.append(al.class_set[0][-1])
			self.baseline_n_top_accs.append(base.class_set[0][-1])

			# plot_top_n_accuracy
			if display_plot_top_n_accuracy or save_plot_top_n_accuracy or display_plot_r2 or save_plot_r2:
				plot_al = plotResults(al)
				plot_base = plotResults(base)

				if display_plot_top_n_accuracy or save_plot_top_n_accuracy:
					plot_al.top_n_accuracy(name = 'alProcess_it' + str(idx_process + 1), folder = 'top_n_accuracy/al/', display = display_plot_top_n_accuracy, save = save_plot_top_n_accuracy)
					plot_base.top_n_accuracy(name = 'base_it' + str(idx_process + 1), folder = 'top_n_accuracy/base/', display = display_plot_top_n_accuracy, save = save_plot_top_n_accuracy)

				if display_plot_r2 or save_plot_r2:
					### TODO : allouer dynamiquement lignes et colonnes
					plot_al.r2(1, 2, name = 'alProcess_it' + str(idx_process + 1), folder = 'r2/al/', display = display_plot_r2, save = save_plot_r2)
					plot_al.r2(1, 2, name = 'base_it' + str(idx_process + 1), folder = 'r2/base/', display = display_plot_r2, save = save_plot_r2)

				del plot_al
				del plot_base

			del al
			del base
			gc.collect()

		# Plot
		plot_hist_n_top_acc(self.alProcess_n_top_accs, self.reg_stra, 'alProcess_n_top_accs', display = display, save = save)
		plot_hist_n_top_acc(self.baseline_n_top_accs, self.reg_stra, 'baseline_n_top_accs', display = display, save = save)


