import os
import pickle
import random
random.seed(42)


if __name__ == "__main__":
    task = "sumo"
    subtasks = ["3by3", "4by4", "5by5"]
    n_roads = [3, 4, 5]
    min_levels = [0.01, 0.01, 0.01]
    max_levels = [0.05, 0.05, 0.05]

    for subtask, n_road, min_level, max_level in zip(subtasks, n_roads, min_levels, max_levels):
        if not os.path.isdir(subtask):
            os.makedirs(subtask)

        task_lambda = []
        for i in range(100):
            task_lambda_temp = []
            with open(f"./{subtask}/train_{i}.rou.xml", "w") as routes:
                print("<routes>", file=routes)
                
                # starts = [f"{direction}_in{j+1}" for j in range(n_direction) for direction in ['E', 'W', 'S', 'N']]
                # ends = [f"{direction}_out{n_direction-j}" for j in range(n_direction) for direction in ['W', 'E', 'N', 'S']]
                starts = [f"road_0_{j+1}_0" for j in range(n_road)] + [f"road_{j+1}_0_1" for j in range(n_road)] + [f"road_{n_road+1}_{j+1}_2" for j in range(n_road)] + [f"road_{j+1}_{n_road+1}_3" for j in range(n_road)]
                ends = [f"road_{n_road}_{j+1}_0" for j in range(n_road)] + [f"road_{j+1}_{n_road}_1" for j in range(n_road)] + [f"road_1_{j+1}_2" for j in range(n_road)] + [f"road_{j+1}_1_3" for j in range(n_road)]
                random_split = [random.random() for _ in range(4)]
                periods = [0.01 + 0.06 * random_split[i] / sum(random_split) for i in range(4)]
                # periods = [min_level + random.random() * (max_level - min_level) for i in range(4)]

                for j, (start, end) in enumerate(zip(starts, ends)):
                    # if start.startswith("E"):
                    #     period = periods[0]
                    # if start.startswith("W"):
                    #     period = periods[0]
                    # if start.startswith("S"):
                    #     period = periods[1]
                    # if start.startswith("N"):
                    #     period = periods[1]
                    period = periods[j // n_road]
                    print(f'\t<flow id="{j}" begin="0.00" from="{start}" to="{end}" end="1800.00" period="exp({period})"/>', file=routes)
                    task_lambda_temp.append(period)
                print("</routes>", file=routes)
                task_lambda.append(task_lambda_temp)

            with open(f"./{subtask}/train_{i}.sumocfg", "w") as simulation:
                print("<configuration>", file=simulation)
                print("\t<input>", file=simulation)
                print(f'\t\t<net-file value="grid.net.xml"/>', file=simulation)
                print(f'\t\t<route-files value="train_{i}.rou.xml"/>', file=simulation)
                print('\t</input>', file=simulation)
                print(f"\t<output>", file=simulation)
                print(f'\t\t<tripinfo-output value="train_{i}.tripinfo.xml"/>', file=simulation)
                print(f"</output>", file=simulation)
                print('</configuration>', file=simulation)

        with open(f"./{subtask}/id_train.pkl", "wb") as f:
            pickle.dump(task_lambda, f)

        task_lambda = []
        for i in range(20):
            task_lambda_temp = []
            with open(f"./{subtask}/valid_{i}.rou.xml", "w") as routes:
                print("<routes>", file=routes)
                
                # starts = [f"{direction}_in{j+1}" for j in range(n_direction) for direction in ['E', 'W', 'S', 'N']]
                # ends = [f"{direction}_out{n_direction-j}" for j in range(n_direction) for direction in ['W', 'E', 'N', 'S']]
                starts = [f"road_0_{j+1}_0" for j in range(n_road)] + [f"road_{j+1}_0_1" for j in range(n_road)] + [f"road_{n_road+1}_{j+1}_2" for j in range(n_road)] + [f"road_{j+1}_{n_road+1}_3" for j in range(n_road)]
                ends = [f"road_{n_road}_{j+1}_0" for j in range(n_road)] + [f"road_{j+1}_{n_road}_1" for j in range(n_road)] + [f"road_1_{j+1}_2" for j in range(n_road)] + [f"road_{j+1}_1_3" for j in range(n_road)]
                random_split = [random.random() for _ in range(4)]
                periods = [0.01 + 0.06 * random_split[i] / sum(random_split) for i in range(4)]
                # periods = [min_level + random.random() * (max_level - min_level) for i in range(4)]

                for j, (start, end) in enumerate(zip(starts, ends)):
                    # if start.startswith("E"):
                    #     period = periods[0]
                    # if start.startswith("W"):
                    #     period = periods[0]
                    # if start.startswith("S"):
                    #     period = periods[1]
                    # if start.startswith("N"):
                    #     period = periods[1]
                    period = periods[j // n_road]
                    print(f'\t<flow id="{j}" begin="0.00" from="{start}" to="{end}" end="1800.00" period="exp({period})"/>', file=routes)
                    task_lambda_temp.append(period)
                print("</routes>", file=routes)
                task_lambda.append(task_lambda_temp)

            with open(f"./{subtask}/valid_{i}.sumocfg", "w") as simulation:
                print("<configuration>", file=simulation)
                print("\t<input>", file=simulation)
                print(f'\t\t<net-file value="grid.net.xml"/>', file=simulation)
                print(f'\t\t<route-files value="valid_{i}.rou.xml"/>', file=simulation)
                print('\t</input>', file=simulation)
                print(f"\t<output>", file=simulation)
                print(f'\t\t<tripinfo-output value="valid_{i}.tripinfo.xml"/>', file=simulation)
                print(f"</output>", file=simulation)
                print('</configuration>', file=simulation)

        with open(f"./{subtask}/id_valid.pkl", "wb") as f:
            pickle.dump(task_lambda, f)

        task_lambda = []
        for i in range(20):
            task_lambda_temp = []
            with open(f"./{subtask}/test_{i}.rou.xml", "w") as routes:
                print("<routes>", file=routes)

                # starts = [f"{direction}_in{j+1}" for j in range(n_direction) for direction in ['E', 'W', 'S', 'N']]
                # ends = [f"{direction}_out{n_direction-j}" for j in range(n_direction) for direction in ['W', 'E', 'N', 'S']]
                starts = [f"road_0_{j+1}_0" for j in range(n_road)] + [f"road_{j+1}_0_1" for j in range(n_road)] + [f"road_{n_road+1}_{j+1}_2" for j in range(n_road)] + [f"road_{j+1}_{n_road+1}_3" for j in range(n_road)]
                ends = [f"road_{n_road}_{j+1}_0" for j in range(n_road)] + [f"road_{j+1}_{n_road}_1" for j in range(n_road)] + [f"road_1_{j+1}_2" for j in range(n_road)] + [f"road_{j+1}_1_3" for j in range(n_road)]
                random_split = [random.random() for _ in range(4)]
                periods = [0.01 + 0.06 * random_split[i] / sum(random_split) for i in range(4)]
                # periods = [min_level + random.random() * (max_level - min_level) for i in range(4)]

                for j, (start, end) in enumerate(zip(starts, ends)):
                    # if start.startswith("E"):
                    #     period = periods[0]
                    # if start.startswith("W"):
                    #     period = periods[0]
                    # if start.startswith("S"):
                    #     period = periods[1]
                    # if start.startswith("N"):
                    #     period = periods[1]
                    period = periods[j // n_road]
                    print(f'\t<flow id="{j}" begin="0.00" from="{start}" to="{end}" end="1800.00" period="exp({period})"/>', file=routes)
                    task_lambda_temp.append(period)
                print("</routes>", file=routes)
                task_lambda.append(task_lambda_temp)

            with open(f"./{subtask}/test_{i}.sumocfg", "w") as simulation:
                print("<configuration>", file=simulation)
                print("\t<input>", file=simulation)
                print(f'\t\t<net-file value="grid.net.xml"/>', file=simulation)
                print(f'\t\t<route-files value="test_{i}.rou.xml"/>', file=simulation)
                print('\t</input>', file=simulation)
                print(f"\t<output>", file=simulation)
                print(f'\t\t<tripinfo-output value="test_{i}.tripinfo.xml"/>', file=simulation)
                print(f"</output>", file=simulation)
                print('</configuration>', file=simulation)

        with open(f"./{subtask}/id_test.pkl", "wb") as f:
            pickle.dump(task_lambda, f)
