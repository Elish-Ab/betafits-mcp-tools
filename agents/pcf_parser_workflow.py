# moved from pcf parser repo
"""
PCF Parser Workflow agent (LangGraph-ready)

Implements the MVP skeleton described by the user:
trigger_input -> fetch_pcf_table -> map_relevant_pcfs -> summarize_matched_pcf ->
create PCF Parser records -> post_process (store to RAG + Graphiti) -> end

This file wires existing `tools.*` utilities and exposes a StructuredTool
`pcf_parser_workflow_tool` for easy use in LangGraph or other orchestrators.
"""

import asyncio
import logging
import json

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from state.state import PCFParserState
from nodes.fetch_pcf_records import fetch_pcf_records_node
from nodes.map_pcfs import map_pcfs_node
from nodes.process_pcfs import process_pcfs_node
from nodes.post_process import post_process_rag_node, post_process_graphiti_node
from chains.tool_definitions import create_tool_from_config

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# =============================================================================
# Workflow Input Schema
# =============================================================================

class WorkflowInput(BaseModel):
    transcript: str = Field(..., description="Meeting transcript or batch messages")
    top_k: int = Field(5, description="Number of PCFs to map (1-5 recommended)")
    type: str = Field(
        "Meeting",
        description="Type of the PCF Parser record (e.g., 'Meeting', 'Context Document', 'Update')",
    )

def run_pcf_parser_workflow(
    transcript: str, top_k: int = 5, type: str = "Meeting"
) -> dict:
    """Run the PCF Parser MVP workflow using LangGraph.

    Returns a dict with keys:
      - created_parser_records: list of created PCF Parser record ids
      - mapped_pcfs: list of mapped PCF ids with scores
      - rag_result: message from RAG storing
      - graphiti_result: True/False or message
    """
    logger.info("=" * 80)
    logger.info("🚀 STARTING PCF PARSER WORKFLOW (LangGraph)")
    logger.info("=" * 80)

    # Initialize StateGraph
    workflow = StateGraph(PCFParserState)

    # Add Nodes
    workflow.add_node("fetch_pcf_records", fetch_pcf_records_node)
    workflow.add_node("map_pcfs", map_pcfs_node)
    workflow.add_node("process_pcfs", process_pcfs_node)
    workflow.add_node("post_process_rag", post_process_rag_node)
    workflow.add_node("post_process_graphiti", post_process_graphiti_node)

    # Define Edges
    workflow.set_entry_point("fetch_pcf_records")
    workflow.add_edge("fetch_pcf_records", "map_pcfs")
    workflow.add_edge("map_pcfs", "process_pcfs")
    workflow.add_edge("process_pcfs", "post_process_rag")
    workflow.add_edge("post_process_rag", "post_process_graphiti")
    workflow.add_edge("post_process_graphiti", END)

    # Compile Graph
    app = workflow.compile()

    # Initial State
    initial_state = {
        "transcript": transcript,
        "top_k": top_k,
        "type": type,
        "pcf_records": [],
        "mapped_pcfs": [],
        "created_parser_records": [],
        "rag_result": None,
        "graphiti_result": None,
        "errors": []
    }

    # Run Graph
    try:
        # Since post_process_graphiti_node is async, the graph execution is async
        final_state = asyncio.run(app.ainvoke(initial_state))
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ WORKFLOW COMPLETE")
        logger.info("=" * 80)
        
        return {
            "created_parser_records": final_state.get("created_parser_records", []),
            "mapped_pcfs": final_state.get("mapped_pcfs", []),
            "rag_result": final_state.get("rag_result"),
            "graphiti_result": final_state.get("graphiti_result"),
        }
    except Exception as e:
        logger.error(f"❌ Workflow failed: {e}")
        return {"error": str(e)}

# =============================================================================
# Create workflow tool from config.json
# =============================================================================

pcf_parser_workflow_tool = create_tool_from_config(
    run_pcf_parser_workflow, "pcf_parser_workflow", WorkflowInput
)

logger.info("✅ Workflow tool created (LangGraph version)")

if __name__ == "__main__":
    # Quick local test harness
    sample_transcript = """
   Project Management Kickoff - 12/24/25
Wed, Dec 24, 2025

0:00 - Matthew Prisco
you're either you've been on the team for a while like Fahad and Afalabi or we have now three new project manager freelancers or project management freelancers and really it's an opportunity not just to add new team members but really to add a new function to our operation and Certainly an important one because I've been doing most of the project management ad hoc, trying to figure out for myself what our project management system might look like, And now I think we really do have not just a foundation for a project management system, but I think really something quite advanced and something well equipped for AI native operations across our entire organization. So it seemed like the right time to bring in some people with project management experience and I think in this first phase we'll want to discuss kind of what what we've already built, what that means for the project management And then we'll also discuss on the call some of the major projects that we have going on. And then we'll see who would be the best of the new freelancers to sort of be that liaison specific to that project. And right now we have three that I hired, one more that I've been talking, with who wasn't gonna be able to make it today. But we'll see kind of you guys would get first pick at which project would be the best fit. And then we can just kind of see if we can get a steady cadence, not just to how the project managers are working, but really how that can help to create more steadiness in the hours and the communication standards and just the overall flow of the existing team members working on their existing projects. So I guess with that maybe it'd be easiest just to give each of the the three of you a chance to introduce yourself and maybe just it doesn't need to be much but just some of your interest and experience in general. You're on mute Vijay.

2:56 - Vijay Jadhav
Sorry guys, I'll start first. So my name is Vijay. I'm working as a project manager from last seven years. I've worked in a variety of sectors such as finance, blockchain, then we can say as a real estate, healthcare, education and all. Recently I've worked on some few of the game development projects as well along with that I've been working as a scrum master for another company which is serious beast for the devops of course so it's like mostly for the devops so I was my responsibilities for them was to kind of the scrums and you know to do the complete project planning so I used to handle that part so yeah mostly my interest is in like finance based applications but even the HR management tools is something I can take care of Thank you guys.

3:54 - Matthew Prisco
Let's go with Ibrahim just because he started talking.

3:57 - Ibrar Hussain
Yeah, sure. Thank you Vijay. That was quite a detailed background that you have shared. I'm glad to hear that you have experience in these domains. So my name is Ibrahim Hussain. I'm working in product management and project management on a dual role from last seven years. And my experience is mostly in SaaS-based products. And I'm working on digital retail solutions for automobile industries of digital financing and asset management, then lead management, then CRM, and then asset financing for lease and finance tools. So majorly, my duties include product roadmapping and then delivering the projects on time with managing the resources, allocating the work, and managing the deadlines that we have decided with the clients. So that's all from my side. And if anybody have any question or any queries, then let's open and welcome all.

5:01 - Matthew Prisco
Very good. Thank you, Ibrahim.

5:02 - Alok Kumar
Hi, everyone.

5:05 - Alok Kumar
So Alok here. So I have around six years of experience as a product manager in domains like investment banking and e-commerce, 3PL, or you can say that for both mobile and web apps. So recently, I worked with one of the clients into the HR domain, where they have managed to streamline the hiring process through one of their app-based, SaaS-based solution. Apart from this, I also work with two clients, like Ready Group Japan, for Japan reason for kind of investments and property selling through digital media. Also, I have experience with e-commerce end to end, like you can say as a 3PL, order management, inventory, or you can say that kind of a SaaS solution on a digital platform, including the payments. So this is all about myself in short.

5:59 - Matthew Prisco
Very good. And then I'll introduce Afalabi and Fahad So Afalabi has been working with us for about a year. He had started with mostly our N8N automations. And as our tech side has grown and as we're now trying to build orchestration systems and more complex, more mature, really, automations, and now that we're starting to shift some of what were no-code tools into coded versions, and GitHub has become more important, really Afolabi's done a lot of the in-the-trenches work with me to sort of validate that what we're building fits together. So I'm really appreciative of the work that he's done, I'm excited that he can add a lot to this project as well, even though his domain isn't project management itself. And then similarly, Fahad's been working with us for the last few months, first as kind of a UX UI designer, but specific for our softer portals that we're now planning to rebuild build in React. So he was familiar with what we'd already built and sort of has been de facto project managing some of what we're working on now. And certainly I would value his feedback for, I'll talk about what I think the main projects are, but certainly one of those is our front tools that can help us really automate not just the conversion of these softer portals into React, but really so that we'll be able to build anything that we need from the front end, but we need it to feel like a process. So I think he may be able to collaborate with whoever becomes sort of the the front-end project manager. The other, I guess, major project is our project system itself. So we will definitely need somebody to sort of project manage our project management tools. So if you feel like you connect with the idea that we have an automation for all of the we're trying to complete and that it may not give us the perfect draft or you know the the human quality but we can get 90 or 95 percent of the way there and that we can create a good PRD from an automation and that we can then create each of the parts of what the project requires, it should really be from the project team, whether that's product manager or project manager, should be in a position to draft whatever it is that we're trying to build. That's kind of the vision for this. And I think when we were doing the introductions, I heard product manager a few times. Certainly project management, I think, is the common thread. And then Concepts like Scrum Master, I think those lean more towards Agile and some of the sprint planning that I'd also like to do. So, you know, let's keep that in mind, kind of that spectrum of product, project, and Agile, and we certainly want to be doing all three. So there's our MCP tools that's becoming a land graph to orchestrate let's say we have a Meeting what can be updated based on the new information from that Meeting if this was like the kickoff for a new project there might be a lot of automations that should trigger we should create the full documentation for the piece which stands for project component feature. The project managers would be involved in reviewing the proposed PCF structure for their project. So obviously the project level is the highest. What components are kind of like a middle organizing layer and then typically we've been calling features any single automation. So it's right now at least it's not a feature in the sense of it does this and this and we'd like it to also do this. It's more anything that is an automation is a feature so that we can organize the documents and the knowledge at the PCF level. Very important. And assign which freelancers or which team members are working on that PCF. So this would fit into, I guess, the core of the project management system, but also that each project manager would have responsibility for reviewing, maintaining the integrity of the PCF structure, what are those components and features of the project, and then what are the objects or artifacts that need to be linked to that PCF based on what type of PCF it is. So that would include things like what are the table that are relevant to that PCF. We call that the data catalog table. Each table has a record there. All of the fields that are a part of that schema are linked to the data catalog record, and we have an automation to update and complete the data catalog and data dictionary record. From the schema that exists in Airtable and from Superbase. So we have at this point a very scalable way to create schemas, review schemas, organize schemas, and that we actually have now a function for data architecture and two freelancers who are primarily functioning as data architects who are reviewing the schema and that the project manager would be involved with at least making sure that the schemas and tables that are relevant to that project are linked and reviewing the overall status. Have they been approved? Do they seem like this is a table that still needs work? And that we would use our data architecture automations and our data architect team in order to make sure that we have what we need there. We create field mappings in a lot of cases. I don't know if that's a standard term or we just kind of started making them, but that allows us to create what we currently call an IO format, but that might be better. Known as an I-O contract. If that project or PCF requires an I-O contract, we can create it with a tool called the field mapper that would be given all of the data catalog and data dictionary records in order to know these are the available fields, what the field mapping needs to be. This is, I think at this point, more of a back-end concept where we've been using it, but I think we could use it similarly for, we have some automations for forms where we're mapping the fields on the forms. We've got, in theory, we could be mapping what are the fields that are displayed in our front end. I'll talk about that in a second. But we have the concept of IO formats that are linked to field mappings that has any configuration that we need. And that, again, our MCP tool automations, when we run it, it can help us create that IO contract that we need. So it would be the project manager who would, as a part of the PRD planning process, a big part of that would be identifying what do we need to create here? What tables is it using? What IO contracts do we need if we're going to generate code? What repo is it using of our existing repos or do we need to create a new repo? If there would be a specific folder that would have that code, we would link that folder or those folders to the PCF. Because we have an automation to take the code in our GitHub and mirror it into Airtable in the repositories, folders, and files tables, and to add other metadata and create the links between those automatically. So it's actually a more insightful layer than just what is natively in GitHub, because we're adding and enforcing our coding engineering standards using what I guess I'm starting to call a context layer. So all of these are context layers that we've created to represent our data architecture, our software, or our coding projects. Similarly for N8N, we have a context layer that would say we're creating an N8n workflow, what are the nodes of that workflow? And if there are LLM nodes that use a prompt, then it links also to the prompts table. Or I'll kind of breeze through this concept, but for Lang Graph, we have the same idea except it takes more tables because you have nodes, edges, edge conditions, chains, and tools, as well as prompts. And that we can represent what we want to build in Lengraph in this context layer format so that we can use the context layer with our code generator tool. And the code generator tool is going to do a much better job than just going to directly to cloud code, chat GPT, whatever tool you want. We actually have the flexibility to use any of those tools, but to use our engineering guidelines and our style guide along with the official context layer and any other context documents that we need. This should be a process of identifying do we have all of the context that we need, can we run an automation, or should an automation have been run to create that? And while we're turning this into a process, probably doing some of this manually, because one of the projects we need to work on is the Lang Graph orchestration of the tools. So I'm hoping that for one of view, that whole concept really resonates of the power of what we can build with that. But I think that's really central to what we're doing. Since I mentioned the front-end format, that's actually also one of the context layers. So for our softer portals, every page gets a record, every block on that page a record and then all of the configurations that we would use to create that design in software. We would draft it first and then build it and then we have an automation to take the HTML and hopefully capture most of what's important about those configurations back into Airtable. My goal is that we will move on from software. There's too many design limitations and what I'm finding now is it takes more time to build with no code than it does to build with code with good AI processes. So I think that the days of no code may be limited, at least for certain types of businesses. But I think we got really a lot of this whole concept because we were doing a lot of stuff with no code, because our data was in Airtable, because we were forcing ourselves to work with software. I think I could appreciate what are the abstraction levels that really that's what no code tools are doing is those are the configurations that are needed to run the code, that they've created, and that really our concept is to define context layers in sort of a no-code-like way, but then we can use those context layers to create what we really want. So, obviously the natural extension on the front end from Softr to React was then a project, in the last two or three weeks, of front-end engineers who we said, we need a process to build what we need in React. And let's first align on what should that context layer be. So in Softr, it was fewer tables, but I think in React, we need more tables. Definitely need more tables. So we have every page, every component, then there's a component tree so that we can align or so we can arrange what are all of the components on the page. If there are buttons or hooks that we need, those have tables. The API and the data layer has a table. So I think we've got consensus from the front-end team about the context layer. Now we have, there are three of them, and one of them is working on the project portal. And another one's working on the prospect portal. And another one's working on the partner portal. Hopefully using the same process, but I think it would make sense to add a project manager with some front-end experience that could hopefully sort of manage the three of them and to make sure that we really are building a front-end process. So Fahad will continue to be involved with that. But I would want to pair him with one of you guys and something that I had noticed actually just today was that I was in contact just one or two messages with Ibrar about the softer portals. And so I assume that means that he has some experience with softer or with front-end development or that I invited him to that project and shouldn't have. But that might be something to take into consideration here if Ibrar is interested in the the front-end project management role that I think that could be a good fit. I don't know, just while I mention that, am I crazy or what do you think of that?

24:19 - Ibrar Hussain
Just a little clarification, when you say the front-end knowledge, it is related to development coding Where are we?

24:29 - Unidentified Speaker
Say that again, I'm sorry.

24:32 - Ibrar Hussain
Yeah, when you asked, did I have any experience related to front end? So I'm just clarifying that when you are asking this experience, experience in coding related stuff engineering?

24:46 - Matthew Prisco
It wouldn't need to be the coding, but I'm just wondering how your profile came up and why I would have invited you for the software project, do you have experience with no-code platforms or software specifically, or do you, like, we wouldn't be using you as the developer, but if you have familiarity with managing front-end projects, then that would be helpful. For that assignment?

25:22 - Ibrar Hussain
Yeah, I do have experience with no-code software, like using AI to develop some kind of software. And so far, it's like Emergent and some other AI tools. But so far, directly involving front-end development are more like I have less experience in this domain. But when it comes to project management, then it's okay for me. For the project management, whether it's front-end, back-end.

25:51 - Matthew Prisco
Okay, well let me give some more details then about what I think is unique about how we would use the front-end process. So, we have our softer documentation or context layer, so we have for each of those portals, if the documentation is complete and up-to-date, we would every page, every block on those pages, and all of the configurations for those blocks. And then we need to sort of convert the functionality of these portals into a React-first design. So each of those, the front-end team documentation, what pages, what components, so instead of software blocks, it's React components, how to arrange those on a component tree, and then adding records on these other tables for things like hooks and API and data uses, and that what we had created for the software portals were documents like a user journey. So the user journey shouldn't change much or at all based on if they're going to be a user of the software portal that we have been trying to build or what the user journey should be for the React version. So we have these user journey documents. And what, in general, the project managers need to do is to make sure that we are organizing these additional, what I call, documents or resources. So, in that case, it would be a Google Doc with the user journey, and then we would be maintaining versions of that document, which could be as PDFs, so one point in time. We would save it as a PDF and continue that to be linked to documents resources. So we have a table called document versions so that we can actually have the version history for documents like that. But for front end, we also have been creating site maps and block designs. So just the concept that there are documents that we would create according to a template that we have or a structure that we want that document to have and that we would organize this so that it can be shared in the port project portal so that the team members can see these documents. So for right now we're sort of going directly into Airtable just making sure everything is linked. Eventually we'll be automations to organize that more consistently and with less direct work. But there's something, I think, unique about this front-end process. And actually, like Fahad had mentioned yesterday, the way we're giving the documentation of the software portal is sort of biasing the drafting of the React version. It's clinging or it's sticking too much to the previous version and the blocks that we needed to use, and it's not designing using what is possible in React. So as a part of this like the, if it's a front-end project, we want to have a standard process prompts and the documents that we're using as context so that we can guide these three other team members who are each building one of these portals. So that's the front-end project manager. Task that we're looking to assign. And then one of the documents and resources that is critical and applies to all of the project management and PCF records is the PRD. So that standard, at this point, the project manager would be maintaining or updating the PRDs. We have an automation, the PRD writer, that we want to now start testing at scale. So each of us will have to organize the PCFs that are a part of that project, make sure that we've run the PCF writer, which gives just the high level details and puts it in Airtable. What's the objective? What are the tools we're using? What is the approach? That we're going to use, not a full PRD, but we need that in Airtable so that it can be displayed in the project portal. And we have an automation called the PCF writer. Then we have another automation called the PRD writer that says these are the sections that we would put in a PRD. This is how each one needs to be structured. And then we have a prompt. That tells it choose the right sections according to the PRD template. So we'll need to see how good are our existing PRDs and how good is the prompt that's giving us those PRDs. So each person would be checking to see do we already have PRDs for each PCF of the project. Make it a better PRD? Or what would align with best practices from other organizations that you've seen? Because I know we would just need to make small changes to our PRD writer tool in order to have the PRD look however we want. Because one of the real, I think, powerful parts of this whole system is every Meeting like we have here is recorded and the MCP LEN graph will process those meetings. We'll find the instructions that are relevant to each PCF and summarize those so that they can be written and stored in a RAG database and in a knowledge graph. We go through the project we will accumulate and organize all of the important details so that they can be used in various other automations but especially or in this case relevant to the PRD writer that we should be able to have the MCP brain can use any of the link documents any of the link data dictionary records, any of the linked IO formats and the field mappings that go along with it, any of the other context layers that we've already created, and any of the project knowledge that's in the knowledge graph in RAG. And it should be able to create a very good PRD. And as we progress through the project, if there have been major changes to to our plan or to the progress overall, we can trigger that PRD writer to do a new draft of the PRD. So at this stage they're still going to be create a PRD manually, review it manually, but I think once we get the MCP tool the project managers will start to see that what we are doing this week, manually, we will be able to actually automate a lot of that. So I mentioned, if I were to try to count what the major projects are, we've got the front-end project. We've got the MCP tools and the project portal. Although the project portal itself would be sort of part of the front-end project, but the project tools and the MCP tools would be displayed. A lot of that data, the PCFs themselves, would be displayed in the project portal. So that one kind of overlaps. The other important one is another project, the CRM Lang Graph. So rather than using HubSpot or Salesforce or anything else, we've got all of our CRM tables are in Airtable right now. And we've built different, I've been calling them domains of the Lang Graph that can be orchestrated with the core CRM brain, but we have like the email pipeline would be, I think I've created it as a component of the CRM LandGraph project. So the email pipeline, every email that we receive, in this case from a prospect, would be processed to link it to the opportunity or create the opportunity if it's a new one. Update core tables like the opportunity or the interaction tables, but really to do the same kind of thing with RAG and Knowledge Graph. So we're using really the same pipeline like we have for the MCP tools on the CRM, but in this case we're organizing the knowledge at the level of the So for the CRM, we organize on the opportunity level, like for MCP tools, we organize at the PCF level. This gives us some similarity in the structure and in the Landgraf code. So we've got these two different projects, the MCP tools Landgraf and CRM Landgraf. We're really trying to get each of developers that in this case back-end developers who's working we have like three or four people on the CRM land graph project they need to be pushing their code to github we need to be reviewing that code for consistency so we have one person Alicia who couldn't make the call but I'll share that the transcript he's been sort of helping helping us aggregate our code, get it all into GitHub. He had helped with the coding standards, the front-end engineering guidelines, back-end engineering guidelines, but we need a little bit more project management of what are the individual contributors working on, what are their blockers from the CRM project. Like I mentioned, we have the email side of this, process every email, a response. We actually took a lot of my emails to train the draft responses. So founder emails is actually a feature of the email pipeline. So you can start to see how there's this PCF hierarchy. And then we need to connect, for example, the email pipeline with the CRM brain so that it can determine what do we need to do after that email was processed. Was the opportunity closed? And one, update the opportunity table. Just because there was an email, we need to update the interaction table. And there are other automations that the CRM brain might need to We've got a lot of web scraping. I call it the, I guess, data enrichment domain, which would be a component. And we've got LinkedIn and Glassdoor and Crunchbase, where those are our main data sources. And we're trying to organize that in the main CRM base and have a flow so that if we add a company to to the CRM that it will enrich the data with what we gather from those platforms. We also have what I call the prospect universe. That is using industry data filings of every company in the United States. So this is like a half a million companies have to file you will report about their benefits. So we have this data is getting stored directly to SuperBase because it's too many records for Airtable. We're adding some fields, performing some additional calculations in SuperBase, and then we need to match. If the company's in our CRM, it needs to be able to fetch the relevant data from SuperBase. Then we need, because we're partnering with VCs, venture capital firms that have invested in startups, fractional CFOs that work with multiple companies on their finance and accounting, and HR consultants, because benefits overlaps with HR, those are good referral partners for us. So we've built the partner portal so that they can see all of their relevant companies so that we can actually extract from Crunchbase what is the whole portfolio for a company, for a venture firm, and then add those companies, match them if they're already in our CRM, and And then run the other enrichment automation so that they can have a nice clean portal. So this is another one of those overlaps where I would want somebody to be the project manager for the CRM LEN graph, but that we would be using that data inside of the partner portal. So if you're really more and have experience with CRM. Obviously there's more to it, but probably I shouldn't get into all of it. That would be an important project and one that I think really would benefit from a good project manager to keep the team on track. Then one last one that I don't think we necessarily need a project manager because off the lobby is working on this more directly on our intake automations and and also on our orchestration engine or what we've sometimes called boat is the acronym I've seen for business orchestration and automation technologies so we need to prepare for okay we've our CRM line graph to fill our pipeline, now we need to give a portal, the prospect portal, so that the company can upload their documents about their current benefits package. We need to process those with different automations for extracting from documents, The intake process is like a project that has a lot of automations, but we're turning this into like a orchestration engine. So it's all config driven. So if right now we're focused on the intake automations, we're also doing our quoting automations using the same processes. And there's like other operations, customer success type automations and what I'm envisioning is we'll use our orchestration engine that is config driven to create any automation using a generic email sender automation. So we have the n8n automation is generic but by pairing it with an IO format and by putting certain configurations on the automation automation project, project steps, and automation steps, that it can give us a more scalable way to create these automations, and certainly a more auditable way to run them. So I think maybe what I'm thinking now, and Aflabi, this aligns with what we've said the last few weeks, is you could kind of be like the project manager while also being an individual contributor for the code. But by being as a part of this team, obviously you've got the most experience with what the project management function might look like, even though we haven't been thinking of it as a project manager. But you can sort of fill that role for those two projects, and we'll get everybody aligned on what would be the right project for them. And what matters to me is that we're trying to do things the same way, that we try to organize the PCFs and link them to other records the same way, that we're using the same automations to create a PCF that's more full or more complete so that we'll be able to display in the project portal, see what PRDs we have, are they good quality or not, maybe make some manual adjustments, but we also need to try to have that turn into adjustments to our prompt and adjustments to our process for how we're creating those. So I think in terms of next steps, it's maybe what we can do it on the call, have a little draft of who would maybe want to do the MCP tools, who would want to do front-end, who would want to do CRM. If you really want to work on a specific project, it's fine that we have two. There's one more, like I said, one more project manager who couldn't make the call. We can, you know, see where to fit him in. Overlapping on set. Does anybody have a specific project that seems the most interesting to them? Alok, when is that one?

46:53 - Alok Kumar
Yeah, so like, for me, I think, like this software part and the CRM part is more interesting, seems interesting for me. So I think I can go with any of them. Either software or the CRM.

47:06 - Matthew Prisco
So that can be- Meaning the front end or the CRM?

47:11 - Vijay Jadhav
Yeah, sure.

47:13 - Matthew Prisco
I can go with CRM.

47:16 - Matthew Prisco
Vijay, you like CRM? Well, that's a really important one because we really want to be able to turn on this CRM lang graph as soon as possible.

47:28 - Alok Kumar
So I can go with software. That's not an issue.

47:34 - Matthew Prisco
Ibrar, since I had already mentioned, you know, the software and the no-code stuff, I don't know if that's that important, if Alok also wants to do that, or what did you think of the kind of MCP tools, the main, this would be more of a back-end system of our project management, We've got maybe four team members working on that including Afolabi and Alicia who I mentioned. Hamna who has done some of the, we actually have a slack bot that we want to turn on so that using the the stored project knowledge if a freelancer has a question It can answer it or make sure that it stays aligned with the project with drafting a response from the project knowledge So that there's some really cool stuff that we're working on on with the the MCP tools land graph but if that sounds interesting that that could be yours and I think that one actually would be a good one to have two people anyway, so I What were you thinking? I don't want to push you in any particular direction.

49:00 - Ibrar Hussain
Yeah, no thank you for clarifying all those. I'm still processing all the information and therefore I'm thinking and I'm in a position to get some more information before choosing anything to go with. So I will give you some text to ask some more information related to those and then I will be in a better position to decide that against which one I should be going with.

49:26 - Matthew Prisco
Okay, that makes sense. Um, well, what I will do is, like I always do, is take the transcript from Read, I'll share the transcript. And the Meeting recording, I don't expect anyone to have to go back to watch the Meeting. Because we have the transcript, you should just be able to load it, ask any questions that you have. So, it's really more about the transcript than about the recording. And I'll do some next steps using ChatGPT. Actually, one thing I should mention on the call is we've been using custom GPTs so that we can load certain documents and kind of create a better baseline for getting your questions answered. Probably do here is make a project management custom GPT, load it with these transcripts, load it with, we'll definitely create like a freelancer project manager guide, like an onboarding kind of document. Here's how we do project management at Betafits. Certainly I'll use the previous meetings for that, the individual that I had with you guys, and that we would load that guide to the custom GPT as well. So one, we'll be creating a custom GPT for project management, but as a project management team, we'll also be evaluating what custom GPTs would our projects need, and where have we created other custom GPTs, and then what documents load in those custom GPTs. I think that's enough for now. Any, any major questions, concerns? Hopefully you guys are excited.

51:29 - Vijay Jadhav
I have a few questions, Matthew.

51:30 - Vijay Jadhav
So when can we actually, you know, have a bit of a actual look at the current system, which is being, you know, developed. So the ready structure in any particular screens that you have developed or any particular document or a part of code or any, you know, if EPAs are there, then we can watch it through or test it through the postman, that kind of stuff.

51:53 - Matthew Prisco
I'm just a bit curious about, you know, it really depends on the project right now and the status of the project. So I'll try to answer that in the context of the CRM. Um, so what we did was we had, um, divided the overall project into these domains. It seems like each domain can be a component and that those components have features. So like I mentioned, the web scrapers, data enrichment is the component and LinkedIn web scraper, Crunchbase web scraper, Glassdoor web scraper are each, features and that we already had the web scraper code. Now we're we were moving it from Selenium to Playwright and that AI was able to assist us with that transition and run it according to the new front-end or I guess back-end Python and back-end engineering standards and Python coding guidelines to it feel more uniform. But for the LandGraph part of it, we had each person submit, here's what their nodes, edges, tools, chains, and prompts need to be. And we did that in a Google Sheet instead of in the database. So we sort of made these files that have never been consolidated into Airtable is where I would have wanted them to be. But I wanted it to be easy for them to make it and not get overwhelmed with the rest of what's in Airtable. So we'll need to review the CRM brain Google Sheet workbook with all of those nodes and the different functions for that. The email pipeline, that person made For the data enrichment and the web scrapers that one's a little bit behind I don't know if they made the draft of the the nodes I think they did but I don't think they started coding it and now that we have these the others who have submitted their code is it in github and Have we reviewed is it consistent at least? With the other side of the project and then can we use the code that we have? And then give the meetings about the the data enrichment the web scrapers the node context layer and just draft the code and give it to that freelancer and say you don't need to draft the code here it is just work from here this should be 90% correct so that's kind of the way we've been doing it but that's what I would view the project manager being in a position to say here's what have, here's what we can use, here's what we can draft, here it is. Tell us if you have any feedback. Okay, it looks good, push it to GitHub. Okay, it's in GitHub, let's bring it into Airtable and run the code reviewer on it and see if it's aligned with the coding guidelines and the engineering standards. For LENGRAF, we need there to be logging as well. So I'm treating the MCP Tools LENGRAF and the CRM LENGRAF as different systems. Even if the architecture is the same, I just don't wanna mix the data. But the logging and state management tables should probably be the same. So we would need to say, are we ready with those tables? Coordinate with the data architecture team. So they're already on these channels. I'm serving as the product, the project manager, probably poorly, but I'm just going from my instincts. So it'd be good to have, you know, like this isn't gonna be a handoff of, you just need to go manage the project from here. I'm probably gonna keep doing a lot of what I've been doing, but you guys will be able to turn that into something that other people can perform and that other people in our processes can improve so that I don't need to feel like I'm just winging it, asking for updates. And, you know, was this done? And I just want to chime in when I notice something on my own. And I want our Slack bot to pretend like it's me so that I don't actually need to keep things aligned. I'm really optimistic about what that will allow us to do. So that's, again, the MCP tools, Lang Graph, working with Hamna to actually deploy what we have so we can turn on this Slack bot and make my life a thousand times easier. So I hope that answered your question.

57:16 - Vijay Jadhav
some parts of it yes yes it does answer my question a bit more clarity of course it's going to be a bit of a bit of a slow transition is going to take a bit of a time but yeah I think we'll catch up on that particular point really quick and maybe you know more and more information we're gonna discuss in this meetings then I think it is of course gonna help us to build the whole platform Alok, I think I had a question back.

57:52 - Alok Kumar
Yeah, I think my question was also get cleared during this explanation. So basically related to the PCA for the project, which I will get a site. I think it is clear for me as well.

58:05 - Matthew Prisco
So again, if you're going to be working on the front end, then Some of the relevant PCFs would be, I think front-end tools is a PCF in the MCP tools project, but that's what has our front-end guidelines and the context layer for React. You would have to review that, make that the context layer is linked to the PCF. That we have, even what we have in Softr, you would have to review the current versions in Softr, what are some of the constraints, like why did we build it a certain way in Softr, and how can we do better with React. And like I said, Fahad's been working on this both as building in software for the prospect portal and then also trying to help align with we have Elijah is is the primary builder right now for the project portal now that we're bringing on more freelancers we want to have an onboarding process through the project portal so that once you start you fill out the intake form like you all did, then we would send your credentials so that at least once your PCFs have been assigned, you will see the PCFs that you've been assigned and then any credentials that you need, if those have been linked to the PCF or linked to you personally. And meetings, as you have the meetings, we process the meetings, those need to be linked to the PCF, maybe also linked to the people who attended the Meeting or who that it's relevant for, so we need to sort of manage what is the, what's the data that is being displayed in these front-end portals, and like are the portals ready, is it because of the front-end or is it because of the data Does that make sense?

1:00:33 - Alok Kumar
So I think it is clear for me as well.

1:00:36 - Matthew Prisco
And what will make that one a little bit challenging is that there are, I guess, the two most important portals right now are the prospect portal and the project, but they're very different. Then we've got also the partner portal, which like I said, that would be for VCs and fractional consultants that are working with a lot of companies. That one's kind of related to the CRM and we may not worry about that one right now. So just to make it easier for you, I would say we should compare what's the status of the project portal in Softr and the status of the project portal in Softr and then what have those freelancers created so far for React and what is the process that they've followed to create that for REACT? Did they follow the same process? Did they follow the process that I've been talking about on all of these front-end meetings? So it will also be looking at the transcripts that we've had for some of those other projects and then see what we can extract from those. And obviously when the MCP Tools is running, that knowledge will also be sorted and stored for the PCF or linked to the PCF in the RAG database and in the Knowledge Graph. Is that clear when I'm saying RAG and Knowledge Graph? Does everyone know what that means? I mean, the Knowledge Graph is new to me, but I think a very powerful one.

1:02:19 - Alok Kumar
Yeah, right. And as you talk about the different So is there any specific place that these processes are documented? Or is this only mapped to the different PCFs directly? So let's say if the documents themselves Yeah, let's say I will be working on the software part as an example. So there are certain processes that the team usually follow in the previous development or for the front end or the back end. So this is somewhere already aligned with the PCFs?

1:02:55 - Unidentified Speaker
When you say aligned, do you mean linked?

1:02:58 - Matthew Prisco
Yeah, yeah. Is it linked?

1:03:00 - Matthew Prisco
We need to see what documents we have. So Alicia will be helpful with this. And Afalabi can also check if we loaded the documents. Like I mentioned, the documents resources table should be mostly or the Google Doc version, like the live version. And then if we want to store version one, version two, those can be PDFs or CSVs that are stored in the document versions table. So I would think that we probably have a lot of these documents are in the documents resources table by now. They should be linked to the correct PCF, once they're linked, then they can be displayed in the project portal. But I'm not exactly sure whether we've processed and organized all of the documents. And certainly, The help with that like the user journeys and We get in the softer version. It's making the react version like softer So Fahad had proposed a new process yesterday. I Had some on that channel that I want us to create a sort of general process for creating what we need in React. And maybe if it says, if you're given other legacy documentation, don't follow it exactly, still follow your React best practices. But definitely Fahad will be able to share more about that. And I'll end up adding you to the front-end team channel. That's really where the project management will take place for that. So you'll be on the project management channel and the front-end team channel. Vijay, you'll be on the project management channel and the CRM land graph channel. Ibrahim will let us know where he fits in. But there's an MCP land graph channel. And then there's also, I guess for a loop, there's also a project portal channel.

1:05:51 - Alok Kumar
Makes sense.

1:05:57 - Matthew Prisco
All right. Well, anybody has anything else? We can go a little bit longer, but I think we covered a lot. Now, Flobby, you have anything to share parting words?

1:06:14 - Afolabi Dave
No, nothing, but just in case, um, you guys may have any questions or you need anything particularly. You can just ask me. I would most likely be available anytime you are. All right.

1:06:35 - Matthew Prisco
All right. Thanks everybody.

1:06:37 - shah_fahad _jalal
Thank you. Thank you, man. Have a good day.
    """
    out = run_pcf_parser_workflow(sample_transcript, top_k=1)
    print("Workflow output:", out)
