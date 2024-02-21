from tensorpc.flow.jsonlike import JsonLikeNode
from tensorpc.core.tree_id import UniqueTreeIdForTree

x = JsonLikeNode(**{
    # "id": UniqueTreeIdForTree("4|root"),
    "id": "4|root",

    "name": "wtf",
    "type": 1
})

print(x.id)