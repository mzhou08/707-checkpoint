from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model, InContextLearningPrompt

import pdb

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

model_path = "liuhaotian/llava-v1.5-7b"
prompt = InContextLearningPrompt(
    prompt="What are the things I should be cautious about when I visit here?",
    examples=[
        [
            "<image-placeholder>",
            "When visiting the pier over the calm lake, there are a few things to be cautious about. First, ensure that you have proper footwear with good traction, as the pier may be wet or slippery, especially if it has been raining or if there is algae on the surface. Second, be mindful of the weather conditions, as sudden changes in weather can make the pier slippery and dangerous. Third, be aware of your surroundings and keep a safe distance from the edge of the pier to avoid accidentally falling into the water. Lastly, if you plan to swim or engage in water activities, be cautious of the water depth and potential hazards, such as submerged rocks or aquatic plants. Always follow safety guidelines and be prepared for any unexpected situations.",
        ],
        [
            "<image-placeholder>",
            "When visiting the desert, it is essential to be prepared for the harsh conditions and potential hazards. Some key considerations include:\n1. Staying hydrated: The desert can be extremely hot, and dehydration can occur quickly. Bring enough water and be mindful of your hydration levels throughout the day.\n2. Protecting your skin: The sun's UV rays can be intense in the desert, increasing the risk of sunburn and skin damage. Wearing sunscreen with a high SPF, sunglasses, and a hat can help protect your skin from harmful UV rays.\n3. Dressing appropriately: Wearing lightweight, breathable, and moisture-wicking clothing is crucial for staying cool and comfortable in the desert. Avoid wearing heavy or bulky clothing that can trap heat and make you feel uncomfortable.\n4. Staying on designated trails: In desert environments, it is essential to stay on marked trails and avoid wandering off into unfamiliar or potentially dangerous areas.\n5. Carrying a first aid kit: In case of an emergency, having a basic first aid kit can be invaluable.\n6. Knowing your limits: The desert can be physically demanding, and it is essential to know your physical abilities and limitations. Avoid overexerting yourself and take breaks when needed.\n7. Being aware of wildlife: Deserts can have various wildlife species, some of which may pose a threat to humans. Be cautious and respectful of the local fauna.\nBy taking these precautions, you can ensure a safe and enjoyable experience in the desert.",
        ],
    ],
    test="<image-placeholder>"
)

image_file = ','.join([
        "https://llava-vl.github.io/static/images/view.jpg",
        "https://cdn.britannica.com/10/152310-050-5A09D74A/Sand-dunes-Sahara-Morocco-Merzouga.jpg",
        "https://environmentamerica.org/wp-content/uploads/2023/05/Flat-Country-credit-Cascadia-Wildlands-DSC_4323.jpeg",
])

# prompt = "<image-placeholder> Rotate Clockwise <image-placeholder> Shift Left <image-placeholder> Rotate Counter-Clockwise <image-placeholder>"
# image_file = ','.join([
#         "/tetris/rot-cw.png",
#         "/tetris/shift-left.png",
#         "/tetris/rot-ccw.png",
#         "/tetris/shift-right.png",
# ])

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

pdb.set_trace()
eval_model(args)
