import numpy as np
import os


def read_conll_f1(filename):
    '''
    This function reads the results of the CoNLL scorer , extracts the F1 measures of the MUS,
    B-cubed and the CEAF-e and calculates CoNLL F1 score.
    :param filename: a file stores the scorer's results.
    :return: the CoNLL F1
    '''
    f1_list = []
    with open(filename, "r") as ins:
        for line in ins:
            new_line = line.strip()
            if new_line.find('F1:') != -1:
                f1_list.append(float(new_line.split(': ')[-1][:-1]))

    muc_f1 = f1_list[1]
    bcued_f1 = f1_list[3]
    ceafe_f1 = f1_list[7]

    return {'MUC': muc_f1, 'B3': bcued_f1, 'CEAFe':ceafe_f1, 'CoNLL F1': (muc_f1 + bcued_f1 + ceafe_f1)/float(3)}


def run_conll_scorer(args):
    # file
    event_response_filename = os.path.join(args.out_dir, 'CD_test_event_mention_based.response_conll')
    event_response_filename = os.path.join(args.out_dir, 'CD_test_event_span_based.response_conll')

    event_conll_file = os.path.join(args.out_dir,'event_scorer_cd_out.txt')

    # command and process
    event_scorer_command = ('perl scorer/scorer.pl all {} {} none > {} \n'.format
        (args.event_gold_file_path, event_response_filename, event_conll_file))

    processes = []
    print('Run scorer command for cross-document event coreference')
    processes.append(subprocess.Popen(event_scorer_command, shell=True))

    while processes:
        status = processes[0].poll()
        if status is not None:
            processes.pop(0)

    print ('Running scorers has been done.')
    print ('Save results...')

    # write from conll file to txt file
    scores_file = open(os.path.join(args.out_dir, 'conll_f1_scores.txt'), 'w')

    event_f1 = read_conll_f1(event_conll_file)
    format_f1 = format_metrics(event_f1)
    scores_file.write('Event F1: {}\n'.format(format_f1))

    scores_file.close()

    return format_f1

def format_conll(metrics):
    """Format metrics for output."""
    result = "MUC: {:.2f} | ".format(metrics['MUC'])
    result += "B3: {:.2f} | ".format(metrics['B3'])
    result += "CEAFe: {:.2f} | ".format(metrics['CEAFe'])
    result += "CoNLL F1: {:.2f} | ".format(metrics['CoNLL F1'])
    return result
