import multiprocessing
import time
import pickle

MAX = 10000000  # 10.000.000
x = []
def fillarray():
    for i in range(0, MAX):
        x.append(i)

    print('Array initiated.')


def double_values(process_name, start, end):
    print('[Process {}]:'.format(process_name), end='')
    x_copy = []
    for i in range(start, end):
        # print('\r\t{}/{}'.format(i - start, end - start - 1), end='')
        x_copy.append(x[i]*2)

    print(' done.')

    pickle_out = open("test_pickles/x_copy_{}_to_{}.pickle".format(start, end), "wb")
    pickle.dump(x_copy, pickle_out)
    pickle_out.close()


fillarray()

double_values('p-1', 0, MAX - 1)
#
runs = 1
num_cores = 2
prof_sum = 0
for k in range(0, runs):
    start = int(round(time.time() * 1000))

    processes = []
    for p in range(0, num_cores):
        processes.append(multiprocessing.Process(target=double_values,
                                                 args=('p{}'.format(p),
                                                       int(p * MAX / num_cores),
                                                       int((p + 1) * MAX / num_cores - 1)),))

    for p in range(0, num_cores):
        processes[p].start()

    for p in range(0, num_cores):
        processes[p].join()

    end = int(round(time.time() * 1000))
    prof_sum += end - start

print('double_values in parallel: {} ms'.format(prof_sum / runs))