from play import organise_files_by_scenario




if __name__ == '__main__':
    dump_flags = {'dump_blotters': True, 'dump_lobs': False, 'dump_strats': False,
                          'dump_avgbals': True, 'dump_tape': True}

    organise_files_by_scenario(dump_flags=dump_flags)