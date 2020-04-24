# Absolutely sh*t implementation, but it freaking works after 11+ hours of straight grinding.

import copy

class Animation:
    def __init__(self, start_frame):
        self.start_frame = start_frame

        self.h_parent = None
        self.h_right = None
        self.h_left = None

    def __str__(self):
        return str(self.start_frame)


class AnimationQueue:  # min-heap structure for maintaining sorted list of waiting animations
    '''
        Priority queue for animations waiting to be played next. Implemented by min-heap,
        organized by earliest start_frame of Animations. (Perhaps in the future, can reimplement
        this to be an implicit data-structure; i.e. using an array that simulates the bin. tree
        of the heap).

        root: the Animation which is the root of the heap (earliest start frame).
    '''

    def __init__(self):
        self.root = None
        self.leaf = self.root  # leaf node on the bottom level to the most right
        self.next_parent = self.root  # parent node for which next added node will be the child

    def add_node(self, node):  # O(log n), total two main O(log(n)) steps: heapify and bad coding.
        if self.root is None:
            self.root = node
            self.leaf = node
            self.next_parent = node
        else:
            # First we insert it into the next available spot
            node.h_parent = self.next_parent
            self.leaf = node

            if self.next_parent.h_left is None:
                self.next_parent.h_left = node
            else:
                self.next_parent.h_right = node

                # Updating next_parent
                ancestor = self.next_parent
                count = 0
                left_move = False
                while ancestor != self.root:
                    temp = ancestor
                    ancestor = ancestor.h_parent
                    count += 1

                    if ancestor.h_left == temp:  # if we made an upwards right move
                        left_move = True
                        break

                if ancestor == self.root and not left_move:
                    self.next_parent = self.left_most_node()
                else:
                    ancestor = ancestor.h_right
                    while count-1 > 0:
                        ancestor = ancestor.h_left
                        count -= 1

                    self.next_parent = ancestor

            self.bubble_up()

    def extract_first(self):
        prev_root = copy.copy(self.root)
        prev_leaf = copy.copy(self.leaf)
        is_prev_leaf_right = 0

        if self.root == self.leaf:
            self.root = None
        else:
            # Updating next_parent and leaf
            if self.leaf.h_parent.h_right == self.leaf:
                self.leaf = self.leaf.h_parent.h_left
                self.next_parent = self.leaf.h_parent
                is_prev_leaf_right = 1
            else:
                ancestor = self.next_parent
                count = 0
                left_move = False
                while ancestor != self.root:
                    temp = ancestor
                    ancestor = ancestor.h_parent
                    count += 1

                    if ancestor.h_right == temp:  # if we made an upwards left move
                        left_move = True
                        break

                if ancestor == self.root and not left_move:
                    self.leaf = self.right_most_node()
                    self.next_parent = self.left_most_node().h_parent
                else:
                    ancestor = ancestor.h_left
                    while count > 0:
                        ancestor = ancestor.h_right
                        count -= 1

                    self.leaf = ancestor

            # print('e', self.leaf)

            # Replacing the root
            if self.root == self.next_parent:
                self.next_parent = prev_leaf

            if self.root == self.leaf:
                self.leaf = prev_leaf

            self.root = prev_leaf

            if prev_root.h_right == prev_leaf.h_parent.h_right and is_prev_leaf_right:
                self.root.h_right = None
            else:
                self.root.h_right = prev_root.h_right

                if prev_root.h_right is not None:
                    prev_root.h_right.h_parent = self.root

            if prev_root.h_left == prev_leaf.h_parent.h_left and not is_prev_leaf_right:
                self.root.h_left = None
            else:
                self.root.h_left = prev_root.h_left

                if prev_root.h_left is not None:
                    prev_root.h_left.h_parent = self.root

            # Making leaf the new root, and erasing leaf out of existence
            if is_prev_leaf_right:
                prev_leaf.h_parent.h_right = None
            else:
                prev_leaf.h_parent.h_left = None

            self.root.h_parent = None

            self.sift_down()
        return prev_root

    def bubble_up(self):
        node = self.leaf
        while node != self.root and node.start_frame < node.h_parent.start_frame:
            self.switch_nodes(node.h_parent, node)

    def sift_down(self):
        node = self.root

        while not self.is_leaf(node):
            if node.h_right is None:
                if node.start_frame >= node.h_left.start_frame:
                    self.switch_nodes(node, node.h_left)
                    continue
                else:
                    break

            if node.h_left is None:
                if node.start_frame >= node.h_right.start_frame:
                    self.switch_nodes(node, node.h_right)
                    continue
                else:
                    break

            # If we've reached this part, it means node has 2 children.
            smaller = None
            if node.h_left.start_frame <= node.h_right.start_frame:
                smaller = node.h_left
            else:
                smaller = node.h_right

            if node.start_frame >= smaller.start_frame:
                self.switch_nodes(node, smaller)
            else:
                break

    def switch_nodes(self, parent, child):
        # Special nodes: Root, leaf, next_parent
        if parent == self.root:
            self.root = child
        elif child == self.root:
            self.root = parent

        if child == self.leaf:
            self.leaf = parent

        if child == self.next_parent:
            self.next_parent = parent
        elif parent == self.next_parent:
            self.next_parent = child

        # Temporary objects
        parents_left = parent.h_left
        parents_right = parent.h_right
        grandparent = parent.h_parent
        childs_left = child.h_left
        childs_right = child.h_right

        # Grandparent and other ancestor handling
        if grandparent is not None:
            if grandparent.h_left == parent:
                grandparent.h_left = child
            else:
                grandparent.h_right = child

        if childs_right is not None:
            childs_right.h_parent = parent

        if childs_left is not None:
            childs_left.h_parent = parent

        if parents_left == child and parents_right is not None:
            parents_right.h_parent = child
        elif parents_right == child and parents_left is not None:
            parents_left.h_parent = child

        # Actual switch
        parent.h_parent = child
        parent.h_right = childs_right
        parent.h_left = childs_left
        child.h_parent = grandparent

        if parents_left == child:
            child.h_left = parent
            child.h_right = parents_right
        else:
            child.h_right = parent
            child.h_left = parents_left

    # O(log n) stupid methods, but who cares for now
    def left_most_node(self):
        node = self.root
        while node.h_left is not None:
            node = node.h_left

        return node

    def right_most_node(self):
        node = self.root
        while node.h_right is not None:
            node = node.h_right

        return node

    def is_leaf(self, node):
        return node.h_right is None and node.h_left is None


import numpy as np

queue = AnimationQueue()
theor_nodes_in_queue = []
has_space = False
for i in range(1000):
    n = np.random.randint(-100, 100)
    node = Animation(n)

    c = None
    if has_space:
        c = np.random.random_integers(0, 3)
    else:
        c = 0

    if c != 2:
        queue.add_node(node)
        theor_nodes_in_queue.append(node.start_frame)
        print('added', n)
        has_space = True
    else:
        potty = queue.extract_first()
        theor_nodes_in_queue.remove(potty.start_frame)
        print('removed', potty)
        if queue.root == None:
            has_space = False

print()
actual_nodes_on_queue = []
while queue.root is not None:
    n = queue.extract_first()
    actual_nodes_on_queue.append(n.start_frame)

print(sorted(theor_nodes_in_queue) == actual_nodes_on_queue)

print(sorted(theor_nodes_in_queue))
print(actual_nodes_on_queue)