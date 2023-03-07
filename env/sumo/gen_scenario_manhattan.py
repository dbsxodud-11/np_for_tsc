import random
from copy import deepcopy
import xml.dom.minidom
import xml.etree.ElementTree as ET

if __name__ == "__main__":
    tree = ET.parse("./manhattan/manhattan.rou.xml")
    root = tree.getroot()

    vehicles = root.findall("vehicle")
    
    original_blocks = [[] for _ in range(20)]
    for vehicle in vehicles:
        original_blocks[int(vehicle.attrib["depart"]) // 180].append(vehicle)
    
    types = ["train", "valid", "test"]
    num_scenarios = [100, 20, 20]

    for ty, num_scenario in zip(types, num_scenarios):
        for i in range(num_scenario):
            random.seed(i)
            shuffle_blocks = deepcopy(original_blocks)
            random.shuffle(shuffle_blocks)

            doc = xml.dom.minidom.Document()
            root = doc.createElement('routes')
            root.setAttribute('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
            root.setAttribute('xsi:noNamespaceSchemaLocation',
                                'http://sumo.dlr.de/xsd/routes_file.xsd')
            doc.appendChild(root)
            # set general info
            node_vtype = doc.createElement('vType')
            node_vtype.setAttribute('id', 'pkw')
            node_vtype.setAttribute('length', '5.0')
            node_vtype.setAttribute('width', '2.0')
            node_vtype.setAttribute('minGap', '2.5')
            node_vtype.setAttribute('maxSpeed', '11.111')
            node_vtype.setAttribute('accel', '2.0')
            node_vtype.setAttribute('decel', '4.5')
            root.appendChild(node_vtype)

            for j, block in enumerate(shuffle_blocks):
                for vehicle in block:
                    node_vehicle = doc.createElement("vehicle")
                    node_vehicle.setAttribute("id", vehicle.attrib["id"])
                    node_vehicle.setAttribute("depart", str(j * 180 + int(vehicle.attrib["depart"]) % 180))

                    node_route = doc.createElement("route")
                    node_route.setAttribute("edges", vehicle[0].attrib["edges"])

                    node_vehicle.appendChild(node_route)
                    root.appendChild(node_vehicle)

            sumofile = f"./manhattan/{ty}_{i}.rou.xml"
            fp = open(sumofile, 'w')
            doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")

        tree = ET.parse("./manhattan/manhattan.sumocfg")
        root = tree.getroot()

        for i in range(num_scenario):
            root[0][1].attrib["value"] = f"{ty}_{i}.rou.xml"

            tree.write(f"./manhattan/{ty}_{i}.sumocfg")
