#include <UIKit/UIKit.h>

@interface VideoProcessor: NSObject
- (NSDictionary *) extractThumbnails:(NSString *)videoUri timestamps:(NSArray *)timestamps;
@end
