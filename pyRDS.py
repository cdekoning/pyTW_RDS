from pyrec.TofDaqRecConnector import get_tofdaqrec
import pyrec.DataStructures as prds
import pyrec.DaqExceptions as prex

#This script assumes a local instance of TofDaqRec is running - no acquisition active
#start the script by getting an instance of tofdaqrec - API functions will be accessible as methods of the
# tofdaqrec object
tdr = get_tofdaqrec()

# Try reading available sources
rds = tdr.rds_get_all_sources()

print(rds)
