import unittest


class testJavaVsPythonMagpie(unittest.TestCase):
    # def test_feature_values(self):
    #     # Load java feature values.
    #     file_name = "new_features_java.txt"
    #     java_features = pd.read_csv(file_name)
    #
    #     # Compute python feature values.
    #     # Load input file containing compositions of materials.
    #     file_name = "new_input_python.txt"
    #     entries = []
    #     with open(file_name, 'r') as f:
    #         for line in f.readlines():
    #             entry = CompositionEntry(composition=line.strip())
    #             entries.append(entry)
    #
    #     # For generating Ward et al. features.
    #     lookup_path = "../../lookup-data/"
    #     sg = StoichiometricAttributeGenerator()
    #     eg = ElementalPropertyAttributeGenerator(use_default_properties=True)
    #     vg = ValenceShellAttributeGenerator()
    #     ig = IonicityAttributeGenerator()
    #
    #     # Generate features.
    #     f_sg = sg.generate_features(entries)
    #     f_eg = eg.generate_features(entries, lookup_path)
    #     f_vg = vg.generate_features(entries, lookup_path)
    #     f_ig = ig.generate_features(entries, lookup_path)
    #
    #     # Concatenate them all.
    #     python_features = pd.concat([f_sg, f_eg, f_vg, f_ig], axis=1)
    #
    #     np_tst.assert_array_equal(java_features.columns,
    #                               python_features.columns)
    #     np_tst.assert_array_almost_equal(java_features.values,
    #                                      python_features.values)

    # def test_edges(self):
    #     java_file_name = "java_edges.txt"
    #     java_zero = []
    #     java_dir = []
    #     java_dist = []
    #     with open(java_file_name, 'r') as f:
    #         lines = f.readlines()
    #     for i in range(0, len(lines), 4):
    #         java_zero.append(list(map(float, lines[i + 1].strip().split())))
    #         java_dir.append(list(map(float, lines[i + 2].strip().split())))
    #         java_dist.append(float(lines[i + 3].strip()))
    #
    #     python_file_name = "python_edges.txt"
    #     python_zero = []
    #     python_dir = []
    #     python_dist = []
    #     with open(python_file_name, 'r') as f:
    #         lines = f.readlines()
    #     for i in range(0, len(lines), 4):
    #         python_zero.append(list(map(float, lines[i + 1].strip().split())))
    #         python_dir.append(list(map(float, lines[i + 2].strip().split())))
    #         python_dist.append(float(lines[i + 3].strip()))
    #
    #     dec = 9
    #     for i in range(len(java_zero)):
    #         np_tst.assert_array_almost_equal(java_zero[i], python_zero[i],
    #                                          decimal=dec)
    #         np_tst.assert_array_almost_equal(java_dir[i], python_dir[i],
    #                                          decimal=dec)
    #         self.assertAlmostEquals(java_dist[i], python_dist[i], delta=10 **
    #                                                                     (-dec))

    def test_ccw(self):
        java_file_name = "java_ccw.txt"
        java = []
        with open(java_file_name, 'r') as f:
            # for line in f.readlines():
                # print line.strip().split()
                # for words in line.strip().split():
                    # print words
                    # sys.exit(1)
            java = [line.strip().split() for line in f.readlines()]

        python_file_name = "python_ccw.txt"
        python = []
        with open(python_file_name, 'r') as f:
            python = [line.strip().split() for line in f.readlines()]

        for i in range(len(java)):
            for j in range(3):
                if java[i][j] != python[i][j]:
                    print java[i], python[i]