import objects as obj

class Rtree:
    def __init__(self, node_max_childern, values=[]):
        self.node_max_childern = 4
        self.root = None

        # Create tree from list of values
        if values:
            self.root = self.build(values)

    def build(self, values):
        pass

    def insert(self, value: obj.Box) -> bool:
        """
        Insert a box object into the R-tree.
        """
        
        # Check if root is None
        success = False
        if self.root is None:
            self.root = obj.Node(
                level=0, 
                mbr=value, 
                value=value, 
                space=self.node_max_childern,
                childern=[]
            )
            return True
        
        # Otherwise, start at the root node
        # and traverse the tree to until a level is found with space
        current_node = self.root
        is_leaf = False

        # Loop until current node is a leaf
        while not is_leaf:
            
            # First check if the inserting box can be a child node within the current node
            contained_within_node = self.__is_box_contained(current_node.mbr, value)

            # If contained within the current node & it has space, add it as a child
            if contained_within_node and current_node.space > 0:
                current_node.childern.append(obj.Node(
                    level=current_node.level + 1,
                    mbr=value,
                    value=value,
                    space=self.node_max_childern,
                    childern=[]
                ))
                current_node.space -= 1
                is_leaf = True # Break out of loop
                success = True

        return success

    def array_representation(self):
        """
        Return the R-tree as an array representation.
        """
        arr = self.traverse(self.root)
        return arr
    
    def traverse(self, node: obj.Node):
        """
        Traverse the R-tree and return a array representation of it.
        """
        if node.childern is None or len(node.childern) == 0 or node.space == self.node_max_childern:
            return node.level

        return [self.traverse(child) for child in node.childern]

    def __is_box_contained(self, primary: obj.Box, secondary: obj.Box) -> bool:
        bottemLeftWithin: bool = primary.bl.x <= secondary.bl.x and primary.bl.y <= secondary.bl.y
        bottomRightWithin: bool = primary.tr.x >= secondary.tr.x and primary.tr.y >= secondary.tr.y
        return bottemLeftWithin and bottomRightWithin 

